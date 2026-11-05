from abc import ABC, abstractmethod
import asyncio
import logging
from PIL import Image

import numpy as np
import torch
from tqdm import tqdm
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import cv2
import kornia as Kn
import kornia.feature as KnF

from src.utils import parse_image

logger = logging.getLogger(__name__)


def to_dict(cfg: DictConfig) -> dict:
    if isinstance(cfg, dict):
        return {k: to_dict(v) for k, v in cfg.items()}
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


class EngineTemplate(ABC):
    @abstractmethod
    def _data_processing(self, dataset: Dataset):
        pass

    @abstractmethod
    def _output_processing(self, dataset: Dataset, **kwargs) -> Dataset:
        pass

    def dry_run(self, dataset: Dataset) -> Dataset:
        nrows = min(5, len(dataset))
        return self.run(dataset.select(range(nrows)))
    
    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        pass


class SIFTEngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        # self.model = cv2.SIFT_create()
        self.model = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)

    def _pil_to_array(self, images: list[Image.Image]) -> list[np.ndarray]:
        """
        Convert PIL image to numpy array (grayscale)
        """
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY) for img in images]

    def _get_matches(self, images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get SIFT keypoints and matches between two images with FLANN
        """
        img2, img1 = images  # tgt image is img1, src image is img2

        # Detect keypoints and compute descriptors
        kp1, des1 = self.model.detectAndCompute(img1, None)
        kp2, des2 = self.model.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        search_params = dict(checks=200)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return pts1, pts2

    def _cal_relative_pose(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Calculate relative pose (R, t) from matched points using the essential matrix.
        Returns R, t, and the number of inliers.
        """
        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        except Exception as e:
            logger.warning(f"🚨🚨 Error in cal_relative_pose: {e}. [DEBUG] {pts1.shape=}, {pts2.shape=}. Maybe not enough matches?")
            R, t = None, None

        return R, t

    def _data_processing(self, dataset: Dataset) -> tuple[list[np.ndarray], np.ndarray]:
        list_of_images = dataset["images"]      
        list_of_intrinsic = dataset["intrinsic"]

        list_of_processed_images = [self._pil_to_array(images) for images in list_of_images]
        list_of_processed_intrinsic = [np.asarray(K, dtype=np.float64) for K in list_of_intrinsic]
        return list_of_processed_images, list_of_processed_intrinsic

    def _output_processing(self, dataset: Dataset, outputs: dict[str, list[np.ndarray | None]]) -> Dataset:
        list_of_R = outputs["R"]
        list_of_t = outputs["t"]

        def _get_rp_description(rp_vector: np.ndarray):
            theta, phi, psi, tx, ty, tz = rp_vector
            return [
                "pitch up" if theta > 0 else "pitch down",
                "yaw right" if phi > 0 else "yaw left",
                "roll clockwise" if psi > 0 else "roll counterclockwise",
                "translate right" if tx > 0 else "translate left",
                "translate down" if ty > 0 else "translate up",
                "translate forward" if tz > 0 else "translate backward",
            ]

        output_rp_vector = []
        output_rp_description = []
        raw_response = []
        for R, t in zip(list_of_R, list_of_t):
            if R is None or t is None:
                output_rp_vector.append(None)
                output_rp_description.append(None)
                raw_response.append("Unknown")
            else:
                theta = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                phi = np.degrees(np.arcsin(-R[2, 0]))
                psi = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                translation = t.flatten().tolist()
                tx = translation[0]

                rp_vector = np.array([theta, phi, psi, translation[0], translation[1], translation[2]])
                rp_description = _get_rp_description(rp_vector)

                output_rp_vector.append(rp_vector)
                output_rp_description.append(rp_description)

                if tx < 0 and phi > 0:
                    raw_resp = "Move left while yawing right"
                elif tx > 0 and phi < 0:
                    raw_resp = "Move right while yawing left"
                else:
                    raw_resp = "Unknown"
                raw_response.append(raw_resp)

        # pyarrow cannot convert multi-dim ndarrays -> convert to nested python lists
        def _to_list(x):
            return None if x is None else np.asarray(x).tolist()

        dataset = dataset.add_column("output_R", [_to_list(R) for R in list_of_R])
        dataset = dataset.add_column("output_t", [_to_list(t) for t in list_of_t])
        dataset = dataset.add_column("output_rp_vector", [_to_list(v) for v in output_rp_vector])
        dataset = dataset.add_column("output_rp_description", output_rp_description)
        dataset = dataset.add_column("raw_response", raw_response)
        return dataset

    def run(self, dataset: Dataset) -> Dataset:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "intrinsic" in dataset.column_names, "Dataset must contain 'intrinsic' column"

        list_of_processed_images, list_of_processed_intrinsic = self._data_processing(dataset)
        results = {
            "R": [],
            "t": [],
        }
        for images, K in tqdm(zip(list_of_processed_images, list_of_processed_intrinsic), desc="Processing images", total=len(list_of_processed_images)):
            pts1, pts2 = self._get_matches(images)
            R, t = self._cal_relative_pose(pts1, pts2, K)
            results["R"].append(R)
            results["t"].append(t)

        return self._output_processing(dataset, results)


class LoFTREngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = KnF.LoFTR(pretrained="indoor_new").to(self.device)

    def _pil_to_tensor(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """
        Convert PIL image to torch tensor (grayscale)
        """
        return [Kn.image_to_tensor(cv2.cvtColor(np.array(img.resize((640, 480))), cv2.COLOR_RGB2GRAY), False).squeeze(0).float() / 255.0 for img in images]

    def _data_processing(self, dataset: Dataset) -> tuple[list[torch.Tensor], np.ndarray]:
        list_of_images = dataset["images"]      
        list_of_intrinsic = dataset["intrinsic"]

        list_of_processed_images = [self._pil_to_tensor(images) for images in list_of_images]
        list_of_processed_intrinsic = [np.asarray(K, dtype=np.float64) for K in list_of_intrinsic]
        return list_of_processed_images, list_of_processed_intrinsic

    def _get_matches(self, images: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get LoFTR keypoints and matches between two images
        """
        img2, img1 = images  # tgt image is img1, src image is img2

        with torch.no_grad():
            input_dict = {"image0": img1.unsqueeze(0).to(self.device), "image1": img2.unsqueeze(0).to(self.device)}
            correspondences = self.model(input_dict)
            mkpts0 = correspondences["keypoints0"].cpu().numpy()
            mkpts1 = correspondences["keypoints1"].cpu().numpy()

        return mkpts0, mkpts1

    def _cal_relative_pose(self, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Calculate relative pose (R, t) from matched points using the essential matrix.
        Returns R, t, and the number of inliers.
        """
        try:
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            num_inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
        except Exception as e:
            logger.warning(f"🚨🚨 Error in cal_relative_pose: {e}. [DEBUG] {pts1.shape=}, {pts2.shape=}. Maybe not enough matches?")
            R, t = None, None

        return R, t

    def _output_processing(self, dataset: Dataset, outputs: dict[str, list[np.ndarray | None]]) -> Dataset:
        list_of_R = outputs["R"]
        list_of_t = outputs["t"]

        def _get_rp_description(rp_vector: np.ndarray):
            theta, phi, psi, tx, ty, tz = rp_vector
            return [
                "pitch up" if theta > 0 else "pitch down",
                "yaw right" if phi > 0 else "yaw left",
                "roll clockwise" if psi > 0 else "roll counterclockwise",
                "translate right" if tx > 0 else "translate left",
                "translate down" if ty > 0 else "translate up",
                "translate forward" if tz > 0 else "translate backward",
            ]

        output_rp_vector = []
        output_rp_description = []
        raw_response = []
        for R, t in zip(list_of_R, list_of_t):
            if R is None or t is None:
                output_rp_vector.append(None)
                output_rp_description.append(None)
                raw_response.append("Unknown")
            else:
                theta = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                phi = np.degrees(np.arcsin(-R[2, 0]))
                psi = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                translation = t.flatten().tolist()
                tx = translation[0]

                rp_vector = np.array([theta, phi, psi, translation[0], translation[1], translation[2]])
                rp_description = _get_rp_description(rp_vector)

                output_rp_vector.append(rp_vector)
                output_rp_description.append(rp_description)

                if tx < 0 and phi > 0:
                    raw_resp = "Move left while yawing right"
                elif tx > 0 and phi < 0:
                    raw_resp = "Move right while yawing left"
                else:
                    raw_resp = "Unknown"
                raw_response.append(raw_resp)

        # pyarrow cannot convert multi-dim ndarrays -> convert to nested python lists
        def _to_list(x):
            return None if x is None else np.asarray(x).tolist()

        dataset = dataset.add_column("output_R", [_to_list(R) for R in list_of_R])
        dataset = dataset.add_column("output_t", [_to_list(t) for t in list_of_t])
        dataset = dataset.add_column("output_rp_vector", [_to_list(v) for v in output_rp_vector])
        dataset = dataset.add_column("output_rp_description", output_rp_description)
        dataset = dataset.add_column("raw_response", raw_response)
        return dataset

    def run(self, dataset: Dataset) -> Dataset:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "intrinsic" in dataset.column_names, "Dataset must contain 'intrinsic' column"

        list_of_processed_images, list_of_processed_intrinsic = self._data_processing(dataset)
        results = {
            "R": [],
            "t": [],
        }
        for images, K in tqdm(zip(list_of_processed_images, list_of_processed_intrinsic), desc="Processing images", total=len(list_of_processed_images)):
            pts1, pts2 = self._get_matches(images)
            R, t = self._cal_relative_pose(pts1, pts2, K)
            results["R"].append(R)
            results["t"].append(t)

        return self._output_processing(dataset, results)


class VLLMOfflineEngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.client = LLM(
            **to_dict(self.model_cfg.vllm_init_kwargs),
        )
    
    def _build_image_content(self, image):
        return {"type": "image_pil", "image_pil": image}

    def _build_text_content(self, text):
        return {"type": "text", "text": text}

    def _build_messages(self, record):
        images = record["images"]
        prompt = record["prompt"]

        image_contents = [self._build_image_content(img) for img in images]
        text_content = self._build_text_content(prompt)

        messages = [{"role": "user", "content": [*image_contents, text_content]}]
        return messages

    def _data_processing(self, dataset: Dataset) -> list[list[dict]]:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "prompt" in dataset.column_names, "Dataset must contain 'prompt' column"

        list_of_messages = [self._build_messages(record) for record in tqdm(dataset, desc="Building messages")]

        return list_of_messages

    def _output_processing(self, dataset: Dataset, outputs: list) -> Dataset:
        output_texts = [output.outputs[0].text for output in outputs]
        return dataset.add_column("raw_response", output_texts)

    def run(self, dataset: Dataset) -> Dataset:
        list_of_messages = self._data_processing(dataset)

        sampling_params = SamplingParams(
            **to_dict(self.model_cfg.sampling_params_kwargs),
        )

        bs = 256
        outputs = []
        for start in tqdm(range(0, len(list_of_messages), bs), desc="Processing batches"):
            logger.info(f"♨️♨️ Processing batch {start} to {min(start + bs, len(list_of_messages))}")
            outputs.extend(self.client.chat(
                list_of_messages[start:start + bs],
                sampling_params=sampling_params,
                use_tqdm=False,
                **to_dict(self.model_cfg.get("vllm_chat_kwargs", {}))
            ))

        return self._output_processing(dataset, outputs)

    def close(self):
        self.client.llm_engine.engine_core.shutdown(timeout=10)
        

class OpenAIResponsesEngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.client = AsyncOpenAI(
            **to_dict(self.model_cfg.sdk_init_kwargs)
        )

    def _build_image_content(self, image):
        res = parse_image(image, "jpeg")
        image_url = res["base64_url"]
        return {"type": "input_image", "image_url": image_url}
    
    def _build_text_content(self, text):
        return {"type": "input_text", "text": text}

    def _build_messages(self, record: dict):
        images = record["images"]
        prompt = record["prompt"]

        image_contents = [self._build_image_content(img) for img in images]
        text_content = self._build_text_content(prompt)

        messages = [{"role": "user", "content": [*image_contents, text_content]}]
        return messages

    def _data_processing(self, dataset: Dataset) -> list[list[dict]]:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "prompt" in dataset.column_names, "Dataset must contain 'prompt' column"

        list_of_messages = [self._build_messages(record) for record in tqdm(dataset, desc="Building messages")]

        return list_of_messages

    def _output_processing(self, dataset: Dataset, outputs: list) -> Dataset:
        output_texts = [output.output_text for output in outputs]
        return dataset.add_column("raw_response", output_texts)

    def run(self, dataset: Dataset) -> Dataset:
        list_of_messages = self._data_processing(dataset)

        async def run_async():
            semaphore = asyncio.Semaphore(12)
            pbar = tqdm(total=len(list_of_messages), desc="Processing requests")

            async with AsyncOpenAI(**to_dict(self.model_cfg.sdk_init_kwargs)) as client:
                async def bounded_process(messages):
                    async with semaphore:
                        response = await client.responses.create(
                            input=messages,
                            **to_dict(self.model_cfg.sampling_params_kwargs)
                        )
                        pbar.update(1)
                        return response

                tasks = [bounded_process(msgs) for msgs in list_of_messages]
                outputs = await asyncio.gather(*tasks)

            pbar.close()
            return outputs

        outputs = asyncio.run(run_async())
        return self._output_processing(dataset, outputs)


class OpenAIChatCompletionsEngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.client = AsyncOpenAI(
            **to_dict(self.model_cfg.sdk_init_kwargs)
        )

    def _build_image_content(self, image):
        res = parse_image(image, "jpeg")
        image_url = res["base64_url"]
        return {"type": "image_url", "image_url": {"url": image_url}}
    
    def _build_text_content(self, text):
        return {"type": "text", "text": text}

    def _build_messages(self, record):
        images = record["images"]
        prompt = record["prompt"]

        image_contents = [self._build_image_content(img) for img in images]
        text_content = self._build_text_content(prompt)

        messages = [{"role": "user", "content": [*image_contents, text_content]}]
        return messages

    def _data_processing(self, dataset: Dataset) -> list[list[dict]]:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "prompt" in dataset.column_names, "Dataset must contain 'prompt' column"

        list_of_messages = [self._build_messages(record) for record in tqdm(dataset, desc="Building messages")]

        return list_of_messages

    def _output_processing(self, dataset: Dataset, outputs: list) -> Dataset:
        output_texts = [output.choices[0].message.content for output in outputs]
        return dataset.add_column("raw_response", output_texts)

    def run(self, dataset: Dataset) -> Dataset:
        list_of_messages = self._data_processing(dataset)

        async def run_async():
            semaphore = asyncio.Semaphore(30)
            pbar = tqdm(total=len(list_of_messages), desc="Processing requests")

            async with AsyncOpenAI(**to_dict(self.model_cfg.sdk_init_kwargs)) as client:
                async def bounded_process(messages):
                    async with semaphore:
                        response = await client.chat.completions.create(
                            messages=messages,
                            **to_dict(self.model_cfg.sampling_params_kwargs)
                        )
                        pbar.update(1)
                        return response

                tasks = [bounded_process(msgs) for msgs in list_of_messages]
                outputs = await asyncio.gather(*tasks)

            pbar.close()
            return outputs

        outputs = asyncio.run(run_async())
        return self._output_processing(dataset, outputs)


class AnthropicEngine(EngineTemplate):
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.client = AsyncAnthropic(
            **to_dict(self.model_cfg.sdk_init_kwargs)
        )

    def _build_image_content(self, image):
        res = parse_image(image, "jpeg")
        format = res["format"]
        image_base64 = res["base64"]
        return {"type": "image", "source": {"type": "base64", "media_type": f"image/{format.lower()}", "data": image_base64}}
    
    def _build_text_content(self, text):
        return {"type": "text", "text": text}

    def _build_messages(self, record):
        images = record["images"]
        prompt = record["prompt"]

        image_contents = [item for idx, img in enumerate(images) for item in [self._build_text_content(f"Image {idx+1}:"), self._build_image_content(img)]]
        text_content = self._build_text_content(prompt)

        messages = [{"role": "user", "content": [*image_contents, text_content]}]
        return messages

    def _data_processing(self, dataset: Dataset) -> list[list[dict]]:
        assert "images" in dataset.column_names, "Dataset must contain 'images' column"
        assert "prompt" in dataset.column_names, "Dataset must contain 'prompt' column"

        list_of_messages = [self._build_messages(record) for record in tqdm(dataset, desc="Building messages")]

        return list_of_messages

    def _output_processing(self, dataset: Dataset, outputs: list) -> Dataset:
        
        def extract_text_from_output(output):
            parts = []
            for block in output.content:
                if block.type== "text":
                    parts.append(block.text)

            text = "\n".join(parts).strip()
            if not text:
                logger.warning(f"Empty text extracted from output: {output}")
            return text
        
        output_texts = [extract_text_from_output(output) for output in outputs]
        return dataset.add_column("raw_response", output_texts)

    def run(self, dataset: Dataset) -> Dataset:
        list_of_messages = self._data_processing(dataset)

        async def run_async():
            semaphore = asyncio.Semaphore(30)
            pbar = tqdm(total=len(list_of_messages), desc="Processing requests")

            async with AsyncAnthropic(**to_dict(self.model_cfg.sdk_init_kwargs)) as client:
                async def bounded_process(messages):
                    async with semaphore:
                        response = await client.messages.create(
                            messages=messages,
                            **to_dict(self.model_cfg.sampling_params_kwargs)
                        )
                        pbar.update(1)
                        return response

                tasks = [bounded_process(msgs) for msgs in list_of_messages]
                outputs = await asyncio.gather(*tasks)

            pbar.close()
            return outputs

        outputs = asyncio.run(run_async())
        return self._output_processing(dataset, outputs)


class InferEngine:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model_name = self.cfg.models.name
        self.engine_type = self.cfg.models.engine

        match self.engine_type:
            case "sift":
                self.engine = SIFTEngine(self.cfg.models)
            case "loftr":
                self.engine = LoFTREngine(self.cfg.models)
            case "vllm_offline":
                self.engine = VLLMOfflineEngine(self.cfg.models)
            case "openai_responses":
                self.engine = OpenAIResponsesEngine(self.cfg.models)
            case "openai_chat_completions":
                self.engine = OpenAIChatCompletionsEngine(self.cfg.models)
            case "anthropic":
                self.engine = AnthropicEngine(self.cfg.models)
            case _:
                raise ValueError(f"Unsupported engine type: {self.engine_type}")
    
    def dry_run(self, dataset: Dataset) -> Dataset:
        return self.engine.dry_run(dataset)
    
    def run(self, dataset: Dataset) -> Dataset:
        return self.engine.run(dataset)

    def close(self):
        if hasattr(self.engine, "close"):
            self.engine.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Custom dataset
    import io
    import requests

    img1_url = "https://picsum.photos/id/237/300"
    img2_url = "https://picsum.photos/id/238/300"

    img1_bytes = requests.get(img1_url).content
    img2_bytes = requests.get(img2_url).content

    img1 = Image.open(io.BytesIO(img1_bytes))
    img2 = Image.open(io.BytesIO(img2_bytes))

    images = [img1, img2]
    prompt = "Compare the two images and describe the differences."

    demo_records = {
        "images": [images for _ in range(10)],
        "prompt": [prompt for _ in range(10)],
    }

    data_hf = Dataset.from_dict(demo_records)

    # Debug
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="main")
    
    engine = InferEngine(cfg)
    output_hf = engine.dry_run(data_hf)
    logger.info(f"🌟🌟 Finishing dry-run")
    engine.close()

    # Check
    for idx, row in enumerate(output_hf):
        logger.info(f"🌀🌀 Record {idx}: {row['raw_response']}")
    # print(f"{output_hf['raw_response']=}")

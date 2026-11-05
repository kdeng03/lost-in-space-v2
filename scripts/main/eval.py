from pathlib import Path
import logging
from src.logging.logging_config import setup_logging
setup_logging()

import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset, concatenate_datasets
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.infer import InferEngine

logger = logging.getLogger(__name__)


def to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def prepare_vrrpi_bench(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"
    assert "demo_prompt" in raw_dataset.column_names, "Dataset must contain 'demo_prompt' column"

    def preprocess(row):
        images = row["images"]
        prompt = row["demo_prompt"]

        image_shapes = [image.size for image in images]
        return {
            "image_shapes": image_shapes,
            "prompt": prompt
        }

    raw_dataset = raw_dataset.map(preprocess)
    return raw_dataset


def quick_eval_vrrpi_bench(dataset: Dataset):
    gt_ans_texts = dataset["gt_ans_text"]
    raw_responses = dataset["raw_response"]
    valid_answers = [
        "Move left while yawing right",
        "Move right while yawing left",
    ]

    import re

    def parse_ans(raw_response):
        ans_region = raw_response[-50:]
        label1, label2 = "Move left while yawing right", "Move right while yawing left"
        if re.search(label1, ans_region):
            return label1
        elif re.search(label2, ans_region):
            return label2
        else:
            return "Unknown"

    ans_texts = [parse_ans(raw_response) for raw_response in raw_responses]

    # Check validity
    is_output_valid = [ans in valid_answers for ans in ans_texts]
    valid_count = sum(is_output_valid)
    total_count = len(is_output_valid)
    validity_rate = valid_count / total_count if total_count else 0.0
    logger.info(
        f"📊📊 Check validity: {valid_count}/{total_count} "
        f"({validity_rate * 100:.2f}%)"
    )

    # Metrics with invalid outputs counted as incorrect.
    accuracy_all = accuracy_score(gt_ans_texts, ans_texts)
    macro_f1_all = f1_score(
        gt_ans_texts,
        ans_texts,
        labels=valid_answers,
        average="macro",
        zero_division=0,
    )
    logger.info(
        f"📊📊 Metrics (invalid as false): "
        f"{accuracy_all=:.4f}, {macro_f1_all=:.4f}"
    )

    # Metrics after dropping invalid outputs.
    valid_indices = [i for i, is_valid in enumerate(is_output_valid) if is_valid]
    valid_gt_ans_texts = [gt_ans_texts[i] for i in valid_indices]
    valid_ans_texts = [ans_texts[i] for i in valid_indices]

    if valid_ans_texts:
        accuracy_valid = accuracy_score(valid_gt_ans_texts, valid_ans_texts)
        macro_f1_valid = f1_score(
            valid_gt_ans_texts,
            valid_ans_texts,
            labels=valid_answers,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"📊📊 Metrics (invalid dropped): "
            f"{accuracy_valid=:.4f}, {macro_f1_valid=:.4f} "
        )
    else:
        logger.info("📊📊 Metrics (invalid dropped): no valid outputs")


def prepare_vrrpi_diag(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"
    assert "demo_prompt" in raw_dataset.column_names, "Dataset must contain 'demo_prompt' column"

    def preprocess(row):
        images = row["images"]
        prompt = row["demo_prompt"]

        image_shapes = [image.size for image in images]
        return {
            "image_shapes": image_shapes,
            "prompt": prompt
        }

    raw_dataset = raw_dataset.map(preprocess)
    return raw_dataset


def quick_eval_vrrpi_diag(dataset: Dataset):
    gt_ans_texts = dataset["gt_ans_text"]
    raw_responses = dataset["raw_response"]
    levels = dataset["level"]
    valid_answer_config = {
        "phi": ["Rotate right", "Rotate left"],
        "theta": ["Rotate up", "Rotate down"],
        "psi": ["Rotate clockwise", "Rotate counterclockwise"],
        "tx": ["Translate right", "Translate left"],
        "ty": ["Translate down", "Translate up"],
        "tz": ["Translate forward", "Translate backward"],
    }

    import re

    def parse_ans(raw_response, level):
        ans_region = raw_response[-50:]
        label1, label2 = valid_answer_config[level]
        if re.search(label1, ans_region):
            return label1
        elif re.search(label2, ans_region):
            return label2
        else:
            return "Unknown"

    ans_texts = [parse_ans(raw_response, level) for raw_response, level in zip(raw_responses, levels)]

    # Check validity
    is_output_valid = [ans in valid_answer_config[level] for ans, level in zip(ans_texts, levels)]
    valid_count = sum(is_output_valid)
    total_count = len(is_output_valid)
    validity_rate = valid_count / total_count if total_count else 0.0
    logger.info(
        f"📊📊 Check validity: {valid_count}/{total_count} "
        f"({validity_rate * 100:.2f}%)"
    )

    # Metrics with invalid outputs counted as incorrect.
    accuracy_all = accuracy_score(gt_ans_texts, ans_texts)
    from collections import defaultdict

    level_f1_scores = defaultdict(float)
    for level, labels in valid_answer_config.items():
        level_indices = [i for i, item_level in enumerate(levels) if item_level == level]
        level_gt_ans_texts = [gt_ans_texts[i] for i in level_indices]
        level_ans_texts = [ans_texts[i] for i in level_indices]
        level_f1_scores[level] = f1_score(
            level_gt_ans_texts,
            level_ans_texts,
            labels=labels,
            average="macro",
            zero_division=0,
        )

    macro_f1_all = sum(level_f1_scores.values()) / len(level_f1_scores)
    logger.info(
        f"📊📊 Metrics (invalid as false): "
        f"{accuracy_all=:.4f}, {macro_f1_all=:.4f}, "
        f"{level_f1_scores=}"
    )  

    # Metrics after dropping invalid outputs.
    valid_indices = [i for i, is_valid in enumerate(is_output_valid) if is_valid]
    valid_gt_ans_texts = [gt_ans_texts[i] for i in valid_indices]
    valid_ans_texts = [ans_texts[i] for i in valid_indices]

    if valid_ans_texts:
        accuracy_valid = accuracy_score(valid_gt_ans_texts, valid_ans_texts)
        level_f1_scores_valid = defaultdict(float)
        for level, labels in valid_answer_config.items():
            level_indices = [i for i in valid_indices if levels[i] == level]
            level_gt_ans_texts = [gt_ans_texts[i] for i in level_indices]
            level_ans_texts = [ans_texts[i] for i in level_indices]
            if level_gt_ans_texts:
                level_f1_scores_valid[level] = f1_score(
                    level_gt_ans_texts,
                    level_ans_texts,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            else:
                level_f1_scores_valid[level] = None

        macro_f1_valid = sum(f for f in level_f1_scores_valid.values() if f is not None) / len(
            [f for f in level_f1_scores_valid.values() if f is not None]
        )
        logger.info(
            f"📊📊 Metrics (invalid dropped): "
            f"{accuracy_valid=:.4f}, {macro_f1_valid=:.4f}, "
            f"{level_f1_scores_valid=}"
        )
    else:
        logger.info("📊📊 Metrics (invalid dropped): no valid outputs")


def _subset(dataset: Dataset, num_per_level: int = 100) -> Dataset:
    """Build a level-balanced subset of the dataset.

    Samples ``num_per_level`` examples from each difficulty level
    (4 levels x 100 = 400 samples by default).
    """
    assert "level" in dataset.column_names, "Dataset must contain 'level' column"
    
    levels = ["15-20", "30-35", "45-50", "60-65"]
    level_subsets = []

    for level in levels:
        level_dataset = dataset.filter(lambda x, level=level: x["level"] == level)
        num_available = len(level_dataset)
        num_select = min(num_per_level, num_available)
        if num_select < num_per_level:
            logger.warning(
                f"⚠️⚠️ Level {level} has only {num_available} samples, "
                f"selecting {num_select} instead of {num_per_level}."
            )
        level_subsets.append(
            level_dataset.shuffle(seed=42).select(range(num_select))
        )

    subset = concatenate_datasets(level_subsets).shuffle(seed=42)
    logger.info(
        f"🌟🌟 Subset built: {len(subset)} samples "
        f"({num_per_level} per level x {len(levels)} levels)"
    )
    return subset


def _resize_images(images: list, size: tuple[int, int]) -> list:
    """Force-resize every image to the exact (width, height) given."""
    return [image.convert("RGB").resize(size, Image.BICUBIC) for image in images]


def prepare_resolution_ablation(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"
    assert "demo_prompt" in raw_dataset.column_names, "Dataset must contain 'demo_prompt' column"

    base_subset = _subset(raw_dataset)

    resolution_config = {
        "tiny": (320, 240),
        "mid": (640, 480),
        "large": (1280, 960),
        "xlarge": (1920, 1440),
    }

    def build_variant(level: str, size: tuple[int, int]) -> Dataset:
        def preprocess(row):
            images = _resize_images(row["images"], size)
            return {
                "images": images,
                "image_shapes": [image.size for image in images],
                "prompt": row["demo_prompt"],
                "resolution_level": level,
            }

        variant = base_subset.map(preprocess)
        logger.info(
            f"🌟🌟 Resolution {level}: {size[0]}x{size[1]}, {len(variant)} samples"
        )
        return variant

    ablation_dataset = concatenate_datasets(
        [build_variant(level, size) for level, size in resolution_config.items()]
    )
    logger.info(
        f"🌟🌟 Resolution ablation dataset built: {len(ablation_dataset)} samples "
        f"({len(base_subset)} base samples x {len(resolution_config)} resolutions)"
    )
    return ablation_dataset


def quick_eval_resolution_ablation(dataset: Dataset):
    import re
    from collections import defaultdict

    levels = dataset["resolution_level"]
    gt_ans_texts = dataset["gt_ans_text"]
    raw_responses = dataset["raw_response"]
    valid_answers = [
        "Move left while yawing right",
        "Move right while yawing left",
    ]

    def parse_ans(raw_response):
        ans_region = raw_response[-50:]
        label1, label2 = "Move left while yawing right", "Move right while yawing left"
        if re.search(label1, ans_region):
            return label1
        elif re.search(label2, ans_region):
            return label2
        else:
            return "Unknown"

    ans_texts = [parse_ans(raw_response) for raw_response in raw_responses]

    # Group by resolution level and report per-level metrics.
    std_acc = defaultdict(float)
    std_f1 = defaultdict(float)
    for level in sorted(set(levels)):
        indices = [i for i, item_level in enumerate(levels) if item_level == level]
        level_gt_ans_texts = [gt_ans_texts[i] for i in indices]
        level_ans_texts = [ans_texts[i] for i in indices]
        is_valid = [ans in valid_answers for ans in level_ans_texts]
        validity_rate = sum(is_valid) / len(is_valid) if is_valid else 0.0

        std_acc[level] = accuracy_score(level_gt_ans_texts, level_ans_texts)
        std_f1[level] = f1_score(
            level_gt_ans_texts,
            level_ans_texts,
            labels=valid_answers,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"📊📊 [{level}] n={len(indices)}, "
            f"validity={validity_rate * 100:.2f}%, "
            f"accuracy={std_acc[level]:.4f}, macro_f1={std_f1[level]:.4f}"
        )

    mean_acc = sum(std_acc.values()) / len(std_acc)
    mean_f1 = sum(std_f1.values()) / len(std_f1)
    logger.info(
        f"📊📊 Metrics (invalid as false): "
        f"{mean_acc=:.4f}, {mean_f1=:.4f}"
    )

    # Metrics after dropping invalid outputs.
    for level in sorted(set(levels)):
        indices = [i for i, item_level in enumerate(levels) if item_level == level]
        valid_indices = [i for i in indices if ans_texts[i] in valid_answers]
        if not valid_indices:
            logger.info(f"📊📊 [{level}] Metrics (invalid dropped): no valid outputs")
            continue
        level_gt_ans_texts = [gt_ans_texts[i] for i in valid_indices]
        level_ans_texts = [ans_texts[i] for i in valid_indices]
        accuracy_valid = accuracy_score(level_gt_ans_texts, level_ans_texts)
        macro_f1_valid = f1_score(
            level_gt_ans_texts,
            level_ans_texts,
            labels=valid_answers,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"📊📊 [{level}] Metrics (invalid dropped): "
            f"n={len(valid_indices)}, "
            f"{accuracy_valid=:.4f}, {macro_f1_valid=:.4f}"
        )


def prepare_reversal_ablation(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"
    assert "demo_prompt" in raw_dataset.column_names, "Dataset must contain 'demo_prompt' column"

    base_subset = _subset(raw_dataset)

    reversal_ablation_config = ["origin", "swap_image", "swap_option"]

    # origin
    def origin(row):
        return {
            "image_shapes": [image.size for image in row["images"]],
            "reversal_level": "origin",
            "prompt": row["demo_prompt"],
        }

    # swap image (2 images)
    def swap_image(row):
        images = row["images"]
        if len(images) < 2:
            raise ValueError("Not enough images to swap.")
        swapped_images = [images[1], images[0]]
        return {
            "images": swapped_images,
            "image_shapes": [image.size for image in swapped_images],
            "reversal_level": "swap_image",
            "prompt": row["demo_prompt"],
        }

    # swap option
    def swap_option(row):
        prompt = row["demo_prompt"]
        _placeholder = "__LABEL_A__"
        swapped_prompt = prompt.replace("Move left while yawing right", _placeholder)
        swapped_prompt = swapped_prompt.replace("Move right while yawing left", "Move left while yawing right")
        swapped_prompt = swapped_prompt.replace(_placeholder, "Move right while yawing left")
        return {
            "image_shapes": [image.size for image in row["images"]],
            "reversal_level": "swap_option",
            "prompt": swapped_prompt,
        }

    origin_subset = base_subset.map(origin)
    swap_image_subset = base_subset.map(swap_image)
    swap_option_subset = base_subset.map(swap_option)

    ablation_dataset = concatenate_datasets(
        [origin_subset, swap_image_subset, swap_option_subset]
    )
    logger.info(
        f"🌟🌟 Reversal ablation dataset built: {len(ablation_dataset)}"
    )
    return ablation_dataset


def quick_eval_reversal_ablation(dataset: Dataset):
    import re
    from collections import defaultdict

    levels = dataset["reversal_level"]
    gt_ans_texts = dataset["gt_ans_text"]
    raw_responses = dataset["raw_response"]
    label1, label2 = "Move left while yawing right", "Move right while yawing left"
    valid_answers = [label1, label2]
    flip = {label1: label2, label2: label1}

    def parse_ans(raw_response):
        ans_region = raw_response[-50:]
        if re.search(label1, ans_region):
            return label1
        elif re.search(label2, ans_region):
            return label2
        else:
            return "Unknown"

    ans_texts = [parse_ans(raw_response) for raw_response in raw_responses]

    # Effective ground truth per reversal level:
    # - origin: unchanged
    # - swap_image: images are swapped, so the correct answer flips
    # - swap_option: only option text positions in the prompt are swapped,
    #   the correct answer text is unchanged
    def effective_gt(gt, level):
        return flip[gt] if level == "swap_image" else gt

    eff_gt_ans_texts = [
        effective_gt(gt, level) for gt, level in zip(gt_ans_texts, levels)
    ]

    # Group by reversal level and report per-level metrics.
    std_acc = defaultdict(float)
    std_f1 = defaultdict(float)
    for level in sorted(set(levels)):
        indices = [i for i, item_level in enumerate(levels) if item_level == level]
        level_gt_ans_texts = [eff_gt_ans_texts[i] for i in indices]
        level_ans_texts = [ans_texts[i] for i in indices]
        is_valid = [ans in valid_answers for ans in level_ans_texts]
        validity_rate = sum(is_valid) / len(is_valid) if is_valid else 0.0

        std_acc[level] = accuracy_score(level_gt_ans_texts, level_ans_texts)
        std_f1[level] = f1_score(
            level_gt_ans_texts,
            level_ans_texts,
            labels=valid_answers,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"📊📊 [{level}] n={len(indices)}, "
            f"validity={validity_rate * 100:.2f}%, "
            f"accuracy={std_acc[level]:.4f}, macro_f1={std_f1[level]:.4f}"
        )

    mean_acc = sum(std_acc.values()) / len(std_acc)
    mean_f1 = sum(std_f1.values()) / len(std_f1)
    logger.info(
        f"📊📊 Metrics (invalid as false): "
        f"{mean_acc=:.4f}, {mean_f1=:.4f}"
    )

    # Metrics after dropping invalid outputs.
    for level in sorted(set(levels)):
        indices = [i for i, item_level in enumerate(levels) if item_level == level]
        valid_indices = [i for i in indices if ans_texts[i] in valid_answers]
        if not valid_indices:
            logger.info(f"📊📊 [{level}] Metrics (invalid dropped): no valid outputs")
            continue
        level_gt_ans_texts = [eff_gt_ans_texts[i] for i in valid_indices]
        level_ans_texts = [ans_texts[i] for i in valid_indices]
        accuracy_valid = accuracy_score(level_gt_ans_texts, level_ans_texts)
        macro_f1_valid = f1_score(
            level_gt_ans_texts,
            level_ans_texts,
            labels=valid_answers,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"📊📊 [{level}] Metrics (invalid dropped): "
            f"n={len(valid_indices)}, "
            f"{accuracy_valid=:.4f}, {macro_f1_valid=:.4f}"
        )


def prepare_ea_i(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"

    def map_func(row):
        images = row["images"]
        rela = row["qa"]["gt_text"]
        gt_idx = row["qa"]["gt_idx"]
        image_shapes = images.size
        return {
            "images": [images],
            "image_shapes": [image_shapes],
            "rela": rela,
            "gt_idx": str(gt_idx),
            "prompt": row["qa"]["prompt"],
        }

    dataset = raw_dataset.map(map_func)

    from datasets import Image, Sequence
    dataset = dataset.cast_column("images", Sequence(feature=Image()))

    return dataset


def quick_eval_ea_i(dataset: Dataset):
    raw_responses = dataset["raw_response"]
    gt_ans_texts = [str(row["qa"]["gt_idx"]) for row in dataset]

    def parse_text(raw_response):
        import re

        ans_region = raw_response[-10:]
        if re.search(r"\d", ans_region):
            return str(re.search(r"\d", ans_region).group(0))
        else:
            return "Unknown"

    ans_texts = [parse_text(raw_response) for raw_response in raw_responses]

    # Check validity
    is_output_valid = [ans.isdigit() for ans in ans_texts]
    valid_count = sum(is_output_valid)
    total_count = len(is_output_valid)
    validity_rate = valid_count / total_count if total_count else 0.0
    logger.info(
        f"📊📊 Check validity: {valid_count}/{total_count} ({validity_rate:.2%})"
    )

    # Metrics with invalid outputs counted as incorrect.
    accuracy_all = accuracy_score(gt_ans_texts, ans_texts)

    logger.info(
        f"📊📊 Metrics (invalid as false): {accuracy_all:.2%} "
    )

    # Metrics after dropping invalid outputs.
    valid_indices = [i for i, is_valid in enumerate(is_output_valid) if is_valid]
    valid_gt_ans_texts = [gt_ans_texts[i] for i in valid_indices]
    valid_ans_texts = [ans_texts[i] for i in valid_indices]

    if valid_ans_texts:
        accuracy_valid = accuracy_score(valid_gt_ans_texts, valid_ans_texts)
        logger.info(
            f"📊📊 Metrics (invalid dropped): {accuracy_valid:.2%} "
        )
    else:
        logger.info("📊📊 Metrics (invalid dropped): no valid outputs")


def prepare_ea_ii(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"

    pe = ["w/o ref obj", "w/ ref obj"]

    def map_func(row, pe):
        images = row["images"]
        image_shapes = [image.size for image in images]
        prompt = row["qa"][pe]["prompt"]
        gt_ans_text_list = row["qa"][pe]["gt_list"]
        return {
            "image_shapes": image_shapes,
            "gt_ans_text_list": gt_ans_text_list,
            "prompt_strategy": pe,
            "prompt": prompt,
        }

    data_list = []
    for prompt_strategy in pe:
        data_list.append(raw_dataset.map(lambda row, ps=prompt_strategy: map_func(row, ps)))

    dataset = concatenate_datasets(data_list)

    return dataset


def quick_eval_ea_ii(dataset: Dataset):
    pass


def prepare_ea_iii(cfg: DictConfig) -> Dataset:
    raw_dataset = load_dataset(**to_dict(cfg.eval.load_dataset_kwargs))

    assert "images" in raw_dataset.column_names, "Dataset must contain 'images' column"

    pe_config = {
        "ZS": "zero-shot",
        "R": "H",
        "R+GT": "H+",
        "R+GT+Rule": "H++",
    }

    def map_func(row, pe):
        pe_key = pe_config[pe]
        images = row["images"]
        image_shapes = [image.size for image in images]
        prompt = row["qa"][pe_key]["prompt"]
        gt_idx = row["qa"][pe_key]["gt_idx"]
        return {
            "image_shapes": image_shapes,
            "prompt_level": pe,
            "gt_idx": str(gt_idx),
            "prompt": prompt,
        }

    data_list = []
    for prompt_strategy in pe_config.keys():
        data_list.append(raw_dataset.map(lambda row, ps=prompt_strategy: map_func(row, ps)))

    dataset = concatenate_datasets(data_list)

    return dataset


def quick_eval_ea_iii(dataset: Dataset):
    import re

    raw_responses = dataset["raw_response"]
    gt_indexes = dataset["gt_idx"]
    prompt_levels = dataset["prompt_level"]
    labels = ["0", "1"]

    def parse_text(raw_response):
        ans_region = raw_response[-20:]
        match = re.search(r"<ans>(\d)</ans>", ans_region) or re.search(r"(\d)", ans_region)
        return match.group(1) if match else "Unknown"

    ans = [parse_text(raw_response) for raw_response in raw_responses]

    def compute_metrics(indices):
        gt = [gt_indexes[i] for i in indices]
        pred = [ans[i] for i in indices]
        accuracy = accuracy_score(gt, pred)
        macro_f1 = f1_score(gt, pred, labels=labels, average="macro", zero_division=0)
        return accuracy, macro_f1

    def report(tag, indices):
        total = len(indices)
        valid_indices = [i for i in indices if ans[i].isdigit()]
        validity_rate = len(valid_indices) / total if total else 0.0

        # Invalid outputs counted as incorrect.
        accuracy_all, macro_f1_all = compute_metrics(indices)
        msg = (
            f"📊📊 [{tag}] n={total}, "
            f"validity={len(valid_indices)}/{total} ({validity_rate:.2%}) | "
            f"invalid as false: acc={accuracy_all:.4f}, macro_f1={macro_f1_all:.4f} | "
        )

        # Invalid outputs dropped.
        if valid_indices:
            accuracy_valid, macro_f1_valid = compute_metrics(valid_indices)
            msg += f"invalid dropped: acc={accuracy_valid:.4f}, macro_f1={macro_f1_valid:.4f}"
        else:
            msg += "invalid dropped: no valid outputs"
        logger.info(msg)

    # Per prompt level (keeps the order in which levels appear in the dataset).
    for level in dict.fromkeys(prompt_levels):
        indices = [i for i, item_level in enumerate(prompt_levels) if item_level == level]
        report(level, indices)

    # Overall
    report("overall", list(range(len(ans))))


def prepare_dataset(cfg: DictConfig) -> Dataset:
    if cfg.eval.name == "vrrpi_bench":
        return prepare_vrrpi_bench(cfg)
    elif cfg.eval.name == "vrrpi_diag":
        return prepare_vrrpi_diag(cfg)
    elif cfg.eval.name == "resolution_ablation":
        return prepare_resolution_ablation(cfg)
    elif cfg.eval.name == "reversal_ablation":
        return prepare_reversal_ablation(cfg)
    elif cfg.eval.name == "ea_i":
        return prepare_ea_i(cfg)
    elif cfg.eval.name == "ea_ii":
        return prepare_ea_ii(cfg)
    elif cfg.eval.name == "ea_iii":
        return prepare_ea_iii(cfg)
    elif cfg.eval.name == "vrrpi_bench_cv":
        return prepare_vrrpi_bench(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.eval.name}")


def quick_eval(cfg: DictConfig, dataset: Dataset):
    if cfg.eval.name == "vrrpi_bench":
        quick_eval_vrrpi_bench(dataset)
    elif cfg.eval.name == "vrrpi_diag":
        quick_eval_vrrpi_diag(dataset)
    elif cfg.eval.name == "resolution_ablation":
        quick_eval_resolution_ablation(dataset)
    elif cfg.eval.name == "reversal_ablation":
        quick_eval_reversal_ablation(dataset)
    elif cfg.eval.name == "ea_i":
        quick_eval_ea_i(dataset)
    elif cfg.eval.name == "ea_ii":
        quick_eval_ea_ii(dataset)
    elif cfg.eval.name == "ea_iii":
        quick_eval_ea_iii(dataset)
    elif cfg.eval.name == "vrrpi_bench_cv":
        quick_eval_vrrpi_bench(dataset)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.eval.name}")


@hydra.main(config_path="../../configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logger.info("🚀🚀 Starting evaluation...")

    # Prepare data
    dataset = prepare_dataset(cfg)
    logger.info("🌟🌟 Data prepared.")

    # Prepare inference engine
    engine = InferEngine(cfg)
    logger.info(f"🌟🌟 Inference engine prepared: {engine.model_name}")

    # Dry run
    dry_run_output_dataset = engine.dry_run(dataset)
    for i, row in enumerate(dry_run_output_dataset):
        logger.info(f"🌀🌀 Sample {i} (dry run): {row['raw_response']}")

    try:
        quick_eval(cfg, dry_run_output_dataset)
    except Exception as e:
        logger.error(f"❌❌ Quick evaluation failed during dry run: {e}")
    logger.info("🌟🌟 Dry run completed.")

    # Run
    output_dataset = engine.run(dataset)
    try:
        quick_eval(cfg, output_dataset)
    except Exception as e:
        logger.error(f"❌❌ Quick evaluation failed during main run: {e}")
    logger.info("🌟🌟 Run completed.")

    # Save
    output_dir = Path(cfg.eval.output_dir)
    if not output_dir.exists():
        logger.info(f"🔧🔧 Output directory {output_dir} does not exist. Creating it...")
        output_dir.mkdir(parents=True, exist_ok=True)

    model_name = engine.model_name
    output_file = output_dir / f"{model_name}.parquet"

    output_dataset.to_parquet(output_file)
    logger.info(f"💾💾 Output dataset saved to {output_file}")
    logger.info("🌟🌟 Output dataset saved.")

    # Close
    engine.close()
    logger.info("✅✅ Evaluation completed successfully.")


if __name__ == "__main__":
    main()

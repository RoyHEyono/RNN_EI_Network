"""Generate one independent W&B sweep config per experiment arm."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
EPSILONS = (0.0, 0.25, 0.5, 0.75)
TASKS = {
    "brightness": "fashionmnist",
    "contrast": "fashionmnist_contrast",
}


def epsilon_text(epsilon: float) -> str:
    return f"{epsilon:g}"


def epsilon_tag(epsilon: float) -> str:
    return epsilon_text(epsilon).replace(".", "p")


def main() -> None:
    output_dir = ROOT / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    for normalization in ("ln", "paramln"):
        template = (ROOT / f"sweep_{normalization}.yaml.template").read_text()
        for task_name, dataset in TASKS.items():
            for epsilon in EPSILONS:
                rendered = (
                    template.replace("__TASK__", task_name)
                    .replace("__DATASET__", dataset)
                    .replace("__EPSILON__", epsilon_text(epsilon))
                    .replace("__EPSILON_TAG__", epsilon_tag(epsilon))
                )
                if "__" in rendered:
                    raise RuntimeError("unresolved placeholder in sweep template")
                filename = (
                    f"sweep_{normalization}_{task_name}_eps"
                    f"{epsilon_tag(epsilon)}.yaml"
                )
                (output_dir / filename).write_text(rendered)
                print(output_dir / filename)


if __name__ == "__main__":
    main()

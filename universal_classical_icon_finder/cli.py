import argparse
from pathlib import Path
from config import DetectionSettings, METHODS
from detector import detect
from detector.output import annotate, results_json, results_csv
from detector.preprocessing import load_rgb
from icon_manager import ProfileManager


def main():
    parser = argparse.ArgumentParser(description="Offline classical UI icon finder")
    parser.add_argument("screenshot", type=Path)
    parser.add_argument("--profile", default="generic")
    parser.add_argument("--icons", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["template_matching", "edge_matching"])
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--scale-step", type=float, default=0.125)
    parser.add_argument("--theme", choices=["auto", "light", "dark"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    try:
        settings = DetectionSettings(tuple(args.methods), args.threshold, args.scale_min, args.scale_max,
                                     args.scale_step, theme=args.theme)
        rgb = load_rgb(args.screenshot)
        report = detect(rgb, ProfileManager().load(args.profile), args.icons, settings)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(results_json(report.results), encoding="utf-8")
        (args.output / "results.csv").write_text(results_csv(report.results), encoding="utf-8")
        annotate(rgb, report.results).save(args.output / "annotated.png")
        print(f"{len(report.results)} matches in {report.timing_ms['total']:.1f} ms. Output: {args.output.resolve()}")
        for warning in report.warnings:
            print(f"Note: {warning}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

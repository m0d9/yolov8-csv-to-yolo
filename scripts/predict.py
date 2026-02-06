#!/usr/bin/env python3
"""Run YOLOv8 inference on images or video.

Usage:
    python scripts/predict.py --weights artifacts/best.pt --source image.jpg
    python scripts/predict.py --weights artifacts/best.pt --source images/ --conf 0.5
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict.py --weights artifacts/best.pt --source image.jpg
  python scripts/predict.py --weights artifacts/best.pt --source images/ --conf 0.5
  python scripts/predict.py --weights artifacts/best.pt --source video.mp4 --save
        """,
    )
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, directory, or video")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save annotated results (default: True)")
    parser.add_argument("--save-dir", type=str, default="runs/predict",
                        help="Directory to save results")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save results")
    args = parser.parse_args()

    save = not args.no_save

    print("=" * 60)
    print("  YOLOv8 Inference")
    print("=" * 60)
    print(f"  Weights: {args.weights}")
    print(f"  Source:  {args.source}")
    print(f"  Conf:    {args.conf}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Save:    {save}")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=save,
        project=args.save_dir,
        name="results",
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print(f"  Inference complete! Processed {len(results)} images.")
    if save:
        print(f"  Results saved to: {args.save_dir}/results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

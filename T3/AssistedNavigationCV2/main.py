"""
Entry point of the Assisted Navigation System
"""

import argparse
from navigation_system import NavigationSystem
from config import CONFIG


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sistema de Navegação Assistida com Visão Computacional"
    )

    parser.add_argument(
        "--video-source",
        type=str,
        help="Fonte de vídeo (0 para webcam, caminho para arquivo de vídeo)",
    )

    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()

    # Load configurations
    config = CONFIG

    # Override video source if provided via command line
    if args.video_source is not None:
        # Convert to int if it's a digit (webcam index)
        if args.video_source.isdigit():
            config.video.source = int(args.video_source)
        else:
            config.video.source = args.video_source
            config.video.source = 0

        print(f"✅ Fonte de vídeo definida via linha de comando: {config.video.source}")

    # Create and run system
    system = NavigationSystem(config)
    system.run()


if __name__ == "__main__":
    main()

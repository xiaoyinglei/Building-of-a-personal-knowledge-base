from pkp.interfaces._ui.cli import app, set_container_factory


def main() -> None:
    app()


__all__ = ["app", "main", "set_container_factory"]


if __name__ == "__main__":
    main()

try:
    from importlib.metadata import version

    __version__ = version("src")
except Exception:
    # Fallback for local development
    __version__ = "0.0.0.dev0"

uv_install_deps:
	uv sync --frozen --all-extras --no-install-project
uv_install_deps_with_upgrade:
	uv sync --all-extras --no-install-project -U
uv_install_deps_compile_with_upgrade:
	uv sync --all-extras --no-install-project -U --compile --no-cache
uv_show_deps:
	uv pip list
uv_show_deps_tree:
	uv tree
uv_create_venv:
	uv venv --python 3.13

pre_commit_install: .pre-commit-config.yaml
	uv run pre-commit install
pre_commit_run: .pre-commit-config.yaml
	uv run pre-commit run --all-files
pre_commit_rm_hooks:
	uv run pre-commit uninstall

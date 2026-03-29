from __future__ import annotations

import subprocess
import sys

import grail_metabolism
import grail_metabolism.__main__ as package_main
import grail_metabolism.experiments as experiments
import grail_metabolism.model as model_api
import grail_metabolism.utils as utils_api
import grail_metabolism.workflows as workflows_api

from grail_metabolism.cli import build_parser, main


def test_top_level_lazy_exports_are_resolvable():
    assert grail_metabolism.MolFrame is utils_api.MolFrame
    assert grail_metabolism.Filter is model_api.Filter
    assert grail_metabolism.load_dataset_bundle is workflows_api.load_dataset_bundle
    assert grail_metabolism.get_experiment_preset is experiments.get_experiment_preset


def test_unknown_lazy_export_raises_attribute_error():
    try:
        getattr(grail_metabolism, "does_not_exist")
    except AttributeError:
        pass
    else:
        raise AssertionError("Unknown package export should raise AttributeError")


def test_module_entrypoint_reexports_cli_main():
    assert package_main.main is main


def test_cli_rules_command_reads_custom_file(tmp_path, capsys):
    rules_path = tmp_path / "rules.txt"
    rules_path.write_text("[C:1]>>[C:1]\n")

    exit_code = main(["rules", "--rules", str(rules_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[C:1]>>[C:1]" in captured.out


def test_cli_parser_contains_supported_commands():
    parser = build_parser()
    subparser_action = next(action for action in parser._actions if getattr(action, "choices", None))
    commands = set(subparser_action.choices)
    assert {"predict", "rules", "presets", "run-preset", "run-config", "ablate", "infer"} <= commands


def test_wrapper_script_run_predict_still_works(tmp_path):
    rules_path = tmp_path / "rules.txt"
    rules_path.write_text("[CH2:1][OH:2]>>[CH:1]=[O:2]\n")

    completed = subprocess.run(
        [sys.executable, "scripts/run_predict.py", "CCO", "--rules", str(rules_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "CC=O" in completed.stdout

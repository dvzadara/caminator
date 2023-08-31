from pathlib import Path
from os import system


project_folder = Path(__file__).resolve().parent.parent
ui_folder = project_folder / "ui"
python_ui_folder = project_folder / "ui_to_py"


def convert_file(input_file, output_file):
    print(f"pyuic5 {input_file} -o {output_file}")
    system(f"pyuic5 {input_file} -o {output_file}")


def prepare_file_name(file):
    return f"{file.stem}.py"


def convert_ui_file_to_python():
    if not python_ui_folder.exists():
        return

    if not ui_folder.exists():
        return

    for ui_file in ui_folder.iterdir():
        if not ui_file.is_file():
            continue

        if ui_file.suffix != ".ui":
            continue

        output_file = python_ui_folder / prepare_file_name(ui_file)
        convert_file(ui_file, output_file)


if __name__ == "__main__":
    convert_ui_file_to_python()

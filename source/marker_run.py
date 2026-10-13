"""Portable Excel-to-XDW yellow marker entry point, without OCR/AI imports."""
from pathlib import Path
import argparse
import sys

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))


def choose_files(root, excel, target):
    import tkinter as tk
    from tkinter import filedialog
    window = tk.Tk()
    window.withdraw()
    try:
        if excel is None:
            selected = filedialog.askopenfilename(parent=window, title='抽出結果Excelを選択',
                initialdir=str(root / 'data'), filetypes=[('抽出結果Excel','*.xlsx')])
            if not selected:
                return None, None
            excel = Path(selected)
        if target is None:
            selected = filedialog.askopenfilename(parent=window, title='黄色マーカーを付ける原本XDWを選択',
                initialdir=str(root / 'input'), filetypes=[('DocuWorks文書','*.xdw')])
            if not selected:
                return None, None
            target = Path(selected)
        return excel, target
    finally:
        window.destroy()


def choose_document(documents, target):
    if target.name in documents:
        return target.name
    import tkinter as tk
    from tkinter import messagebox
    selected = []
    window = tk.Tk()
    window.title('Excel内の対応文書を選択')
    tk.Label(window, text=f'対象XDW: {target.name}\nExcel内の対応する原本ファイルを選んでください。',
             padx=20, pady=10).pack()
    listing = tk.Listbox(window, width=72, height=min(12,len(documents)), exportselection=False)
    for name in documents:
        listing.insert(tk.END, name)
    listing.pack(padx=20,pady=10)
    def accept():
        choice = listing.curselection()
        if not choice:
            messagebox.showinfo('文書の選択','対応する文書を選択してください。', parent=window)
            return
        selected.append(documents[choice[0]])
        window.destroy()
    tk.Button(window, text='この文書の座標を使う', command=accept).pack(pady=10)
    window.mainloop()
    return selected[0] if selected else None


def main():
    parser = argparse.ArgumentParser(description='抽出Excelの座標を使ってXDWコピーへ黄色マーカーを追加')
    parser.add_argument('excel', nargs='?', type=Path)
    parser.add_argument('xdw', nargs='?', type=Path)
    parser.add_argument('--root', type=Path, default=SOURCE.parent)
    parser.add_argument('--document', help='Excelの原本ファイル列にある対応文書名。PDF由来・改名時に指定。')
    parser.add_argument('--self-test', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.self_test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(root/'source/tests'), pattern='test_marker_*.py')
        if suite.countTestCases() == 0:
            raise RuntimeError('追加機能のテストがありません。')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    interactive = args.excel is None or args.xdw is None
    excel, target = choose_files(root,args.excel,args.xdw) if interactive else (args.excel,args.xdw)
    if excel is None or target is None:
        print('キャンセルしました。', flush=True)
        return 0
    from simple_drawing.marker_excel import read_workbook, MarkerInputError
    from simple_drawing.xdw_marker import apply_markers
    workbook = read_workbook(excel)
    document = args.document
    if document is None:
        if target.name in workbook.documents:
            document = target.name
        elif interactive:
            document = choose_document(workbook.documents, target)
            if document is None:
                print('キャンセルしました。', flush=True)
                return 0
        else:
            raise MarkerInputError('対象XDW名に対応する行がありません。--documentでExcel内の原本ファイル名を指定してください。')
    rows = workbook.select(document)
    print(f'Excel: {excel}\n対象XDW: {target}\nExcel内の文書: {document}\n座標のある対象行: {len(rows)}', flush=True)
    from filelock import FileLock, Timeout
    (root/'data').mkdir(exist_ok=True)
    try:
        with FileLock(root/'data/operation.lock',timeout=0):
            output, report = apply_markers(root, excel, target, rows, document=document,
                 skipped_no_coordinates=workbook.skipped_no_coordinates(document),
                 progress=lambda message: print(message, flush=True))
    except Timeout as exc:
        raise RuntimeError('別の処理が実行中です。終了後に再実行してください。') from exc
    if interactive:
        import tkinter as tk
        from tkinter import messagebox
        window=tk.Tk(); window.withdraw()
        messagebox.showinfo('黄色マーカー追加 完了',f'{len(rows)}行の座標へ配置しました。\n\n出力先:\n{output}',parent=window)
        window.destroy()
    return 0


if __name__ == '__main__':
    try:
        code = main()
    except KeyboardInterrupt:
        print('中断しました。原本は変更していません。', flush=True)
        code = 130
    except Exception as exc:
        print('黄色マーカー追加に失敗しました: ' + str(exc), file=sys.stderr, flush=True)
        code = 1
    raise SystemExit(code)

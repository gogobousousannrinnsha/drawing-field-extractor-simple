"""Create genuine yellow Core marker annotations on a new XDW copy."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import ctypes
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

from .marker_excel import MarkerInputError, MarkerRow

MM_PER_PT = 25.4 / 72
MAX_STROKES = 20000


@dataclass(frozen=True)
class Stroke:
    row: MarkerRow
    width_pt: int
    points: tuple[tuple[float, float], tuple[float, float]]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def marker_strokes(row):
    """Cover the coordinate rectangle with horizontal strokes (Viewer 1..13pt)."""
    # Rounding to whole points may extend each edge by at most half a point.
    # Split taller rectangles to preserve the editable marker width range.
    count = max(1, math.ceil(row.height_mm / (13 * MM_PER_PT)))
    if count > MAX_STROKES:
        raise MarkerInputError('座標領域の縦幅が大きすぎます。')
    width_pt = min(13, max(1, math.ceil(row.height_mm / count / MM_PER_PT)))
    thickness = width_pt * MM_PER_PT
    y_start = row.y_mm + min(row.height_mm, thickness) / 2
    y_end = row.y_mm + row.height_mm - min(row.height_mm, thickness) / 2
    # Marker end caps are included in the annotation; inset the center line.
    inset = min(row.width_mm / 2 - 0.005, thickness / 2)
    inset = max(0, inset)
    x1, x2 = row.x_mm + inset, row.x_mm + row.width_mm - inset
    for i in range(count):
        y = y_start if count == 1 else y_start + (y_end - y_start) * i / (count - 1)
        yield Stroke(row, width_pt, ((round(x1, 2), round(y, 2)), (round(x2, 2), round(y, 2))))


def plan(rows, dimensions):
    strokes = []
    for row in rows:
        if row.page not in dimensions:
            raise MarkerInputError(f'Excel {row.sheet_row}行: ページ{row.page}は対象XDWにありません。')
        size = dimensions[row.page]
        if row.x_mm + row.width_mm > size['width_mm'] + 0.02 or row.y_mm + row.height_mm > size['height_mm'] + 0.02:
            raise MarkerInputError(f'Excel {row.sheet_row}行: 座標が対象XDWのページ範囲を超えています。')
        strokes.extend(marker_strokes(row))
        if len(strokes) > MAX_STROKES:
            raise MarkerInputError('マーカーが多すぎます。Excelを文書単位で分けてください。')
    if not strokes:
        raise MarkerInputError('選択文書には座標のある行がありません。')
    return strokes


def page_metadata(doc, number):
    from docuworks_ctypes._raw import types as T
    from docuworks_ctypes.errors import check_result
    info = T.XDW_PAGE_INFO_EX()
    info.nSize = ctypes.sizeof(info)
    check_result(doc.raw.XDW_GetPageInformation(doc.handle, number,
        ctypes.cast(ctypes.byref(info), ctypes.POINTER(T.XDW_PAGE_INFO))), 'XDW_GetPageInformation')
    if info.nDegree % 90 or info.nWidth <= 0 or info.nHeight <= 0:
        raise MarkerInputError(f'ページ{number}のサイズ・回転を取得できません。')
    return {'width_mm': info.nWidth / 100, 'height_mm': info.nHeight / 100,
            'source_rotation': info.nDegree % 360, 'page_type': info.nPageType}


def validate_saved_sources(root, rows, dimensions, target_hash):
    """When local extraction records exist, bind Excel rows to their frozen source."""
    from .storage import local_path, read_json
    from .excel import _payload_regions, _box
    manifests, batches, records = {}, {}, {}
    verified, standalone = [], []
    for row in rows:
        pieces = row.result_id.split(':')
        if len(pieces) != 4 or not re.fullmatch(r'[A-Za-z0-9_-]+',pieces[0]):
            standalone.append(row.sheet_row)
            continue
        run_id, index_text, page_text, ordinal_text = pieces
        folder = root/'data/runs'/run_id
        manifest_path = folder/'run.json'
        if not manifest_path.exists():
            standalone.append(row.sheet_row)
            continue
        try:
            index, page, ordinal = int(index_text), int(page_text), int(ordinal_text)
            if index < 0 or ordinal < 1 or page != row.page:
                raise ValueError('結果IDの文書・ページ・順番が不正です。')
            if run_id not in manifests:
                manifest = read_json(manifest_path)
                if manifest['status'] not in ('complete','partial'):
                    raise ValueError('抽出処理が完了していません。')
                batch_path = local_path(root,manifest['contract']['batch_path'])
                if sha(batch_path) != manifest['contract']['batch_sha256']:
                    raise ValueError('保存された前処理記録のハッシュが一致しません。')
                manifests[run_id], batches[run_id] = manifest, read_json(batch_path)
            manifest, batch = manifests[run_id], batches[run_id]
            document = batch['documents'][index]
            if document['name'] != row.document:
                raise ValueError('Excelと保存記録の原本ファイル名が一致しません。')
            saved_source = local_path(root,document['source'])
            if sha(saved_source) != document['sha256']:
                raise ValueError('保存原本のハッシュが一致しません。')
            if document.get('source_type') == 'xdw' and target_hash != document['sha256']:
                raise ValueError('指定XDWが抽出時の原本と一致しません。抽出に使用したXDWを選んでください。')
            if len(document['pages']) != len(dimensions):
                raise ValueError('指定XDWと抽出元のページ数が一致しません。')
            saved_page = next(p for p in document['pages'] if p['number']==page)
            if any(abs(saved_page[k]-dimensions[page][k])>0.05 for k in ('width_mm','height_mm')):
                raise ValueError('指定XDWと抽出元のページサイズが一致しません。')
            key = (run_id,index,page)
            if key not in records:
                reference = manifest['records'][f'{index}:{page}']
                record_path = local_path(folder,reference['path'])
                if sha(record_path) != reference['sha256']:
                    raise ValueError('保存抽出結果のハッシュが一致しません。')
                record = read_json(record_path)
                records[key] = (record,_payload_regions(folder,record))
            record, regions = records[key]
            result = record['parsed']['results'][ordinal-1]
            if list(row.evidence_ids) != result['source_ids']:
                raise ValueError('Excelと保存抽出結果の根拠IDが一致しません。')
            boxes = [_box(regions[e]) for e in row.evidence_ids]
            if not boxes:
                raise ValueError('保存OCRには根拠座標がありません。')
            x,y=min(b[0] for b in boxes),min(b[1] for b in boxes)
            expected=(x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y)
            if any(abs(a-b)>0.05 for a,b in zip(expected,(row.x_mm,row.y_mm,row.width_mm,row.height_mm))):
                raise ValueError('Excelの座標が保存OCRの根拠座標と一致しません。')
            verified.append(row.sheet_row)
        except (KeyError,ValueError,IndexError,StopIteration,OSError) as exc:
            raise MarkerInputError(f'Excel {row.sheet_row}行の保存根拠を照合できません: {exc}') from exc
    return {'saved_source_verified_rows': verified, 'standalone_rows': standalone}


def _verify(doc, before, planned):
    from docuworks_ctypes.enums import AnnotationType, Color
    grouped = {}
    for stroke in planned:
        grouped.setdefault(stroke.row.page, []).append(stroke)
    checked = []
    for number, expected in grouped.items():
        annotations = list(doc.page(number).annotations(recursive=False))
        if len(annotations) != before[number] + len(expected):
            raise RuntimeError(f'ページ{number}のマーカー保存数が一致しません。')
        for index, stroke in enumerate(expected, start=before[number]):
            annotation = annotations[index]
            if annotation.annotation_type != AnnotationType.MARKER:
                raise RuntimeError('保存された注釈がマーカーではありません。')
            if int(annotation.get_standard_attribute('%BorderColor')) != int(Color.YELLOW):
                raise RuntimeError('保存されたマーカーが黄色ではありません。')
            if annotation.get_standard_attribute('%BorderWidth') != stroke.width_pt:
                raise RuntimeError('保存されたマーカーの線幅が一致しません。')
            if not annotation.get_standard_attribute('%BorderTransparent'):
                raise RuntimeError('保存されたマーカーが透過設定ではありません。')
            actual = annotation.get_standard_attribute('%Points')
            if len(actual) != 2 or any(abs(p.x-x) > 0.02 or abs(p.y-y) > 0.02 for p,(x,y) in zip(actual, stroke.points)):
                raise RuntimeError('保存されたマーカーの座標が一致しません。')
            checked.append({'page': number, 'annotation_index': index + 1,
                            'excel_row': stroke.row.sheet_row, 'width_pt': stroke.width_pt,
                            'points_mm': stroke.points})
    return checked


def apply_markers(root, excel, target, rows, *, document, skipped_no_coordinates=0, progress=print, api_factory=None):
    from docuworks_ctypes import XdwApi
    from docuworks_ctypes.enums import AuthMode, OpenMode, Color
    from docuworks_ctypes.geometry import PointMM

    root, excel, target = Path(root).resolve(strict=True), Path(excel).resolve(strict=True), Path(target).resolve(strict=True)
    if target.suffix.casefold() != '.xdw' or not target.is_file():
        raise MarkerInputError('対象には.xdwファイルを指定してください。')
    rows = list(rows)
    target_hash, excel_hash = sha(target), sha(excel)
    temp_root = root / 'temp'
    temp_root.mkdir(exist_ok=True)
    output_dir = root / 'data' / 'markers' / (datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'excel': str(excel), 'excel_sha256': excel_hash,
              'target': str(target), 'target_sha256': target_hash, 'source_document': document,
              'selected_rows': len(rows), 'skipped_no_coordinates': skipped_no_coordinates,
              'coordinate_system': 'sdk-displayed-page-top-left-mm', 'rows': [asdict(row) for row in rows]}
    report_path = output_dir / 'マーカー処理結果.json'
    final = output_dir / (target.stem + '_黄色マーカー.xdw')
    def record():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    record()
    try:
        api = api_factory() if api_factory else XdwApi.load()
        report['core_version'] = version('docuworks-ctypes')
        report['sdk_version'] = api.runtime_info.version_text
        progress('対象XDWのコピーとページ・座標を確認しています。')
        with tempfile.TemporaryDirectory(prefix='marker-', dir=temp_root) as temporary:
            copy = Path(temporary) / 'document.xdw'
            if len(str(copy).encode('utf-16-le')) // 2 > 255:
                raise MarkerInputError('アプリのフォルダー名が長すぎます。短い場所へ移して実行してください。')
            shutil.copyfile(target, copy)
            if sha(copy) != target_hash:
                raise RuntimeError('XDWのコピーが原本と一致しません。')
            with api.open_document(copy, mode=OpenMode.READONLY, auth=AuthMode.NO_DIALOG) as doc:
                if not 1 <= doc.page_count <= 10000:
                    raise MarkerInputError('対象XDWのページ数が対応範囲外です。')
                dimensions = {n: page_metadata(doc,n) for n in range(1,doc.page_count+1)}
                before = {n: len(list(doc.page(n).annotations(recursive=False))) for n in dimensions}
            planned = plan(rows, dimensions)
            report['provenance'] = validate_saved_sources(root, rows, dimensions, target_hash)
            report.update(pages=dimensions, annotations_before=before, markers=len(planned))
            record()
            with api.open_document(copy, mode=OpenMode.UPDATE, auth=AuthMode.NO_DIALOG) as doc:
                for i, stroke in enumerate(planned, 1):
                    doc.page(stroke.row.page).add_marker([PointMM(x,y) for x,y in stroke.points],
                        border_color=Color.YELLOW, border_width=stroke.width_pt, border_transparent=True)
                    if i == 1 or i % 50 == 0 or i == len(planned):
                        progress(f'黄色マーカーを配置中: {i}/{len(planned)}（ページ{stroke.row.page}）')
                progress('XDWを保存しています。')
                doc.save()
            progress('保存したXDWを再読込みして、マーカーの数・色・座標を確認しています。')
            with api.open_document(copy, mode=OpenMode.READONLY, auth=AuthMode.NO_DIALOG) as doc:
                if doc.page_count != len(dimensions):
                    raise RuntimeError('保存後のページ数が一致しません。')
                added = Counter(stroke.row.page for stroke in planned)
                for number, metadata in dimensions.items():
                    if page_metadata(doc,number) != metadata:
                        raise RuntimeError(f'保存後のページ{number}のサイズ・回転が変わりました。')
                    if len(list(doc.page(number).annotations(recursive=False))) != before[number] + added[number]:
                        raise RuntimeError(f'保存後のページ{number}の注釈数が一致しません。')
                report['verified_markers'] = _verify(doc, before, planned)
            if sha(target) != target_hash or sha(excel) != excel_hash:
                raise RuntimeError('処理中に原本XDWまたはExcelが変更されました。出力を確定できません。')
            # A fresh run directory and same-volume rename prevent partial output.
            os.rename(copy, final)
        report.update(status='complete', output=str(final), output_sha256=sha(final),
                      original_preserved=True, excel_preserved=True)
        record()
        progress(f'完了: {len(rows)}行、{len(planned)}個の黄色マーカー。\n出力: {final}')
        return final, report_path
    except BaseException as exc:
        def unchanged(path, expected):
            try:
                return sha(path)==expected
            except OSError:
                return False
        report.update(status='failed', error=str(exc), original_preserved=unchanged(target,target_hash),
                      excel_preserved=unchanged(excel,excel_hash))
        try:
            record()
        except OSError as log_error:
            exc.add_note('処理結果JSONを保存できませんでした: ' + str(log_error))
        raise

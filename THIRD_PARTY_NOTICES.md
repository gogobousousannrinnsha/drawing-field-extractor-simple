# 第三者資産とライセンス

## この公開ZIPの範囲

このZIPは自作のPythonコード、BAT、取得処理、設定例、説明書、テストコード、取得先とハッシュの情報、ライセンス参照文書と、自作のDocuWorks Core 1.0.1を配布するものです。自作部分の著作権表示とMIT本文は`LICENSE`にあります。既存のMIT版から流用した自作コードの著作権表示も保持しています。

Python、外部の依存パッケージ、第三者DLL、Ollama、モデル、入力図面、OCR結果、実行履歴はZIPの配布対象に含めません。自作部分のMITは、外部から取得するソフトウェア・モデル・規約本文には適用されません。

## 同封するDocuWorks Core 1.0.1

`bundled/core/docuworks_ctypes-1.0.1-py3-none-any.whl`は、[dw-ocr v0.8.1で公開した配布物](https://github.com/gogobousousannrinnsha/dw-ocr/releases/tag/v0.8.1)を変更せずに同封します。MIT、Copyright (c) 2026 gogobousousannrinnshaです。MIT本文・著作権表示・パッケージの権利情報をwheel内に保持します。

固定する配布物は48,173 bytes、SHA-256は`7ea3a6d1623238180ccb400237a753b616dccf72761ba83d97fee1b55f7f5f58`です。取得元と検査情報は`assets/bundled-core.json`に記録し、初期設定ではネット取得せず、同封wheelを検査してフォルダー内runtimeへ導入します。

これはPythonで記述した自作APIラッパーの配布です。DocuWorks本体、富士フイルムのSDK・ヘッダー・DLL、Integrationsは同封しません。DocuWorks製品の使用・SDKの利用には、各製品の条件が別途適用されます。現在のアプリはPDF入力のみで、Coreの同封はXDW入力処理の実装・動作確認を意味しません。

## 初期設定で取得する資産

取得先、固定版、SHA-256は`assets/`のロック情報を参照してください。初期設定は利用者のPCで各配布元から取得し、このアプリのフォルダー内に配置します。取得物のLICENSE・NOTICE・`*.dist-info`等の権利情報を削除せず保持します。

PythonとTcl/Tkは公式の固定版MSI、91のPythonパッケージは固定wheel、OCRの検出・認識モデルはPaddleの公式アーカイブ、Ollamaは公式の固定版ZIPから取得します。Gemma4も専用Ollamaで取得し、固定digestを検査します。公開ZIPには、PP-OCRモデルのApache 2.0参照本文と帰属通知を`licenses/ocr/`へ収録し、初期設定でモデル保存先にも配置します。

| 資産 | 主な条件と確認先 |
|---|---|
| Python | PSFおよび収録された第三者ライセンス。[Python公式](https://docs.python.org/3/license.html) |
| Tcl/Tk | Tcl/Tkの個別の許諾。[Tcl公式](https://www.tcl-lang.org/software/tcltk/license.html) |
| PaddlePaddle、PaddleOCR、PaddleX、PP-OCRモデル | 公式資産のApache 2.0と同梱する第三者条件。[PaddlePaddle](https://github.com/PaddlePaddle/Paddle)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[PaddleX](https://github.com/PaddlePaddle/PaddleX) |
| pypdfium2／PDFium | pypdfium2の許諾とPDFiumの第三者ライセンス。[pypdfium2公式](https://pypdfium2.readthedocs.io/en/stable/readme.html#licensing) |
| pypdf等のPython依存 | パッケージごとの許諾。間接依存を含む版とSHAはwheelロックを参照 |
| Ollama | MITと同梱する第三者ライブラリの条件。[Ollama固定版](https://github.com/ollama/ollama/tree/v0.35.1) |
| Gemma4 | Apache 2.0。[Google公式本文](https://ai.google.dev/gemma/apache_2)、保存参照文書は`licenses/models/gemma4-apache-2.0.txt` |
| Gemma3（任意） | Gemma Terms of UseとProhibited Use Policy。[規約](https://ai.google.dev/gemma/terms)、[禁止用途](https://ai.google.dev/gemma/prohibited_use_policy) |
| Saluki／llama.cpp（追加資産を導入する場合） | モデル・量子化モデル・画像用モデルとエンジンそれぞれの条件。[モデル配布元](https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0)、[llama.cpp](https://github.com/ggml-org/llama.cpp) |
| GPUランタイム | NVIDIA／AMD等の取得物に含まれる個別条件。アプリのMITでは扱わない |

Gemma3の導入、確認、選択、抽出には、保存した現行規約と禁止用途への受諾記録を必要とします。Gemma3およびその派生モデルの使用は、Gemma規約に組み込まれた禁止用途と適用法令の制限を受けます。モデルの利用について、このアプリからそれらに反する許諾を与えることはありません。公式本文、旧版参考資料、指定Noticeと取得由来を`licenses/models/`に保存しています。

Salukiは`Saluki追加設定.bat`による任意取得です。`assets/saluki-assets.json`にモデル、画像用モデル、固定llama.cpp、必要通知の取得先とSHAを記録します。追加設定ではモデル選択を変更せず、取得・配置の検査が完了しても`inference_verified`は`false`にします。SHA一致は資産の同一性の確認であり、推論の確認や、構築済みフォルダー全体の再配布許諾の確認を代用しません。

## 構築後のフォルダーを配布するとき

初期設定後のフォルダーには第三者のコード・DLL・モデルが入ります。そのフォルダーの再配布は、公開ソースZIPとは異なる内容の配布です。Apache／BSD／MIT等の本文と帰属表示、Gemma3の規約とNotice、LGPL等の対応ソース・交換方法、CUDA／cuDNN等の配布条件を、実際のファイルに対応させて確認してください。初期設定で自分用に取得できることだけを、そのまま再配布許諾の証明にしないでください。

入力する図面や画像、取得結果にも、権利者・利用者側の利用条件が適用されます。公開ZIPには利用者の原本・OCR・Excelを含めず、テストで必要な画像は自作コードから生成します。

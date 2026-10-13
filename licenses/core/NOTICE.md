# 同封する自作 Core 1.0.1

`docuworks-ctypes` 1.0.1 を、公開済みの MIT 配布物から変更せず同封します。
Python ソースを含む wheel であり、DocuWorks 本体、DLL、SDK、図面原本は含みません。

- 配布元: [dw-ocr v0.8.1 公開 Release](https://github.com/gogobousousannrinnsha/dw-ocr/releases/tag/v0.8.1)
- ファイル: `docuworks_ctypes-1.0.1-py3-none-any.whl`
- 配置: `bundled/core/`
- 容量: 48,173 bytes
- SHA-256: `7ea3a6d1623238180ccb400237a753b616dccf72761ba83d97fee1b55f7f5f58`
- ライセンス: MIT、Copyright (c) 2026 gogobousousannrinnsha
- 通知: wheel 内の LICENSE を変更せず [LICENSE.txt](LICENSE.txt) にも保存しています。

初期設定はこの同封 wheel をローカルの runtime に導入します。
Core はネットから再取得しません。資産が欠けたり変更された場合は初期設定を失敗として表示します。
他の外部 runtime、ライブラリ、モデルの取得方針は従来どおりです。

本アプリには、同封 Core を使う XDW 入力アダプターと、抽出 Excel から XDW コピーへ黄色マーカーを配置する処理があります。
XDW を実際に開くには、利用者側で対応する正規の DocuWorks 実行環境を用意する必要があります。
Core の MIT ライセンスは、DocuWorks 本体や DLL の再配布権を付与するものではありません。

元の配布物を改変せず保存し、パッケージ名・版・MIT 通知・全 RECORD・資産 SHA の一致を検査しています。

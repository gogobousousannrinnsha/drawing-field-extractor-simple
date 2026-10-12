# ライセンス文書の保存範囲

このフォルダーは、自作コードとは別に適用されるモデル規約の参照資料を含みます。自作部分のMITライセンスは、取得するモデルや第三者の実行資産のライセンスを置き換えません。

`models/sources.json`に、各本文の公式URL、取得日、文書の版、原ページのSHA-256、保存本文のSHA-256を記録しています。本文はGoogleの公式ページから抽出し、HTMLレイアウトと空白をテキスト表示に整えています。条項の意味を編集した翻訳や要約ではありません。

| 文書 | 用途 |
|---|---|
| `models/gemma3-terms.txt` | Gemma3導入時に提示する現行のGemma Terms of Use（2026-04-01） |
| `models/gemma3-prohibited-use-policy.txt` | Gemma3に適用される禁止用途の本文（2024-02-21） |
| `models/gemma3-terms-2024-04-01.txt` | Ollamaの既存モデル内の規約と照合するための旧版参考資料 |
| `models/gemma3-NOTICE.txt` | Gemma指定のNotice |
| `models/gemma4-apache-2.0.txt` | Gemma4のApache License 2.0の参照本文 |

Gemma3の受諾処理では、現行規約と禁止用途の保存本文のSHAを記録します。旧版を保存したことだけで、現行規約への受諾を代用しません。モデルから取得できるライセンスblobも、そのまま保持します。

出典はGoogle AI for Developersの各公式ページです。ページは、別記がある部分を除き[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)と記載されています。著者はGoogle、変更点はHTMLと空白の表示形式のみです。モデル利用の条件は、各モデルの規約・ライセンス本文に従ってください。

初期設定で取得するPython、Paddle関連、Ollama、PDFium等の本文は、取得物の中に含まれるLICENSE・NOTICE・ライセンス情報を保持します。公開ソースZIPは、これらの実行資産やモデル本体を含みません。構築後のフォルダーを他者へ配布するときは、実際の同梱物全体の再配布条件を別途確認する必要があります。

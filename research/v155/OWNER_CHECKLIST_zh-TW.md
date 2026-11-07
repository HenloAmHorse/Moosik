# 1.5.5 prereview：短測試清單

尚未發布，請只跑交付資料夾裡的 prereview exe。它會像一般版本讀寫你的設定。
建議先備份 `C:\Users\PC\.moosik\spectrum.json`；回去用不認得 True CQT 的
舊 exe 時，可關閉播放器後還原這份設定。音樂與既有快取不需搬移或刪除。

1. **既有功能**：先維持平常的 FFT／ASLT 設定播放熟悉的曲子，確認快取照樣命中、
   頻譜／Whole track／歌詞翻譯正常。這輪沒有更換你的模型或改 ASLT 算法。
2. **新方法**：開 Spectrum → Analysis → Analysis method → True CQT。
   應切到 Pre-process，顯示 Resolution／Window／Analysis frame rate，收起 FFT
   padding、overlap、插值等不適用控制；沒有快取時按 Re-analyze。
   有 live PCM 時，等待期間應看到明確標示 LIVE FFT PREVIEW 的動態畫面，
   可展開 Live FFT preview 調大小／視窗，但不應重新啟動 CQT 分析。
   Native DSD／DoP 沒有 live PCM 時應如實顯示無預覽；Whole track 只等正式 CQT。
3. **畫面與時軸**：先用預設 48 bins/octave／Blackman／180 fps，分析一首短歌。
   播放、暫停、seek、重播，並看 Whole track 的縮放、follow、hover。
   CQT 完成時應換掉 FFT 預覽與其 waterfall 歷史，改用儲存的 CQT。
   低音有長視窗造成的時間展寬是方法特性，不代表播放變慢。
4. **左右聲道**：產生 channels 後試 Left／Right／Split／Diff，再回 Mix。
   測試明顯偏一邊的段落；若手邊有反相測試檔，Mix 應抵消而左右各自仍有能量。
5. **快取與設定**：重播同一首不應重新分析或重設 Whole track 倍率。
   關閉重開應命中 CQT 快取；改 BPO／window／fps 要辨認成其他設定。
   切回 FFT／Superlet，再回 True CQT，應記住 CQT 設定且不覆寫 ASLT 的頻率尺度。
6. **資源與取消**：分析時按 Abort，應能稍後再試且不留下可當成完成結果的半份快取。
   高 BPO／低頻可能明確拒絕資源需求；不應偷偷縮短視窗、降設定或換成 FFT。
   分析核心數設定 1／2／6 時，log 的 max parallel slots 不應超出設定，
   0 表示純序列執行，不是沒有用 CPU。
   本版沿用共用設定，目前要先在 Superlet 的分析運算設定調整，再切回 CQT。
   這不是整個程式 CPU 百分比上限，解碼、UI 等其他工作仍會使用 CPU。

有問題時請附：exe 名稱、音檔採樣率／時長、bars、BPO、window、analysis fps、
聲道模式，以及對應 `[analysis]`／`[cqt]` log。效能回報請區分分析完成時間與畫面 FPS。
請先關閉舊播放器再開新版。若要比較分析速度，選尚未有該設定 CQT 快取的曲子；
舊 CQT 快取仍相容，命中快取的瞬間載入不代表本輪分析速度。

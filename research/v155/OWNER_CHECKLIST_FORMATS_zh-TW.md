# 1.5.5 格式支援測試版：短測試清單

請先關閉其他 Moosik，再開交付資料夾裡的新 exe。這是一般播放器設定模式，
**不是**把設定存在 exe 旁邊的 portable mode。它會照常讀寫你的設定；可先備份
`C:\Users\PC\.moosik\spectrum.json`，音樂與既有快取不必搬動或刪除。

1. **新增格式**：若手邊有 Opus（`.opus`、`.ogg`、`.webm`）、AAC-LC／ALAC
   (`.m4a`)、AIFF 或 CAF，用 Add Files 與 Load Folder 各試一次。確認有出現在
   播放清單，Info 的實際 codec／採樣率／聲道／lossless 標示合理，能播放到結束。
   WebM／MP4／MKV 在這裡只匯入音軌，不會顯示影片。不是所有 codec profile 都支援。
2. **跳轉與接續**：試 Opus 的前跳、後跳、暫停後跳轉、重播與切下一首。
   跳轉後應沒有舊片段或明顯時軸錯位。這版 Opus 為正確保留解碼歷史而從頭
   解碼到目標；長檔跳轉可能較慢，不能把它當成已完成快速索引的版本。
3. **分析路徑**：對一個新格式檔跑平常的 ASLT，確認分析完成後 Mix／Left／Right
   與 Whole track 都可用；重開／重播可命中快取。既有 FLAC／WAV／MP3 也照常播放。
4. **True CQT 回歸**：未有相同設定快取時，等待分析期間應顯示明確標示的
   LIVE FFT PREVIEW（需有 live PCM），完成才切成 CQT；再試 Waterfall、左右聲道、
   Whole track 的縮放／follow。Full CQT 的低頻長視窗沒有被偷偷縮短。
5. **原有功能**：照常開歌詞、翻譯、繁簡顯示、頻譜，確認沒有回退到更早的介面。
   本輪沒有更換你的翻譯模型或改 ASLT 的數值定義。

不必為了本輪重跑大量分析。遇到問題請附 exe 名稱、檔案容器與實際 codec、
時長、採樣率、操作順序及對應 log。速度回報請區分「分析完成時間」與畫面 FPS；
命中既有快取的瞬間載入不代表本輪分析變快。

已知範圍：Opus 只新增 mapping-family-0 的 mono／stereo；HE-AAC、WMA、APE、
WavPack 沒有在此輪新增。AAC-LC 可以播放，但 MP4 encoder-delay／edit-list 尾端
修整不宣稱逐樣本精確。完整範圍見交付資料夾的 AUDIO_FORMATS.md。

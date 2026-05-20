#!/bin/bash

# 定義要執行的測試腳本列表
SCRIPTS=(
    "testOddFramesProcession.py"
    "testOddFramesAMS05.py"
    "testOddFramesZombie.py"
    "testOddFramesWalkInPark.py"
)

# 用於儲存結果的陣列
declare -a RESULTS

echo "🚀 開始執行所有測試腳本 (預計處理 4 支影片)..."
echo "--------------------------------------------------"

for SCRIPT in "${SCRIPTS[@]}"; do
    if [ ! -f "$SCRIPT" ]; then
        echo "❌ 錯誤: 找不到檔案 $SCRIPT"
        RESULTS+=("$SCRIPT: 檔案不存在")
        continue
    fi

    echo "正在執行 $SCRIPT ..."
    
    # 執行腳本並抓取最後一行包含 "Calculated BDrate =" 的內容
    # 這裡假設 Case 2 的結果會是輸出中最後一次出現的 BDrate 數值
    BDRATE=$(python "$SCRIPT" 2>&1 | grep "Calculated BDrate =" | tail -n 1 | awk -F'= ' '{print $2}')

    if [ -z "$BDRATE" ]; then
        RESULTS+=("$SCRIPT: 擷取失敗")
        echo "   ⚠️ 無法抓取到 BD-rate。"
    else
        RESULTS+=("$SCRIPT: $BDRATE")
        echo "   ✅ 完成: $BDRATE"
    fi
done

echo ""
echo "=================================================="
echo "         BD-rate (Case 2) 總結報告"
echo "=================================================="
for RES in "${RESULTS[@]}"; do
    echo "$RES"
done
echo "=================================================="
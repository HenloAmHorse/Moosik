#!/bin/sh
# Sequential bass-response matrix through the production pre-process entry.
cd /c/Users/PC/Programming/moosik-v155-claude || exit 1
. /c/Users/PC/Programming/moosik-v155-claude-home/scripts/env.sh
OUT=/c/Users/PC/Programming/moosik-v155-claude-home/raw/bass_response.jsonl
echo "# run $(date -u +%FT%TZ) head $(git rev-parse HEAD) dirty $(git status --porcelain | wc -l)" >> $OUT
for fx in ${FIXTURES:-gate click pair}; do
  for arm in ${ARMS:-aslt1 aslt9 cqt12 cqt48 cqt192}; do
    s=$(date +%s)
    line=$(MOOSIK_GPU=0 MOOSIK_BASS_ARM=$arm MOOSIK_BASS_FIXTURE=$fx cargo test --release --locked --bin moosik bass_response -- --ignored --nocapture --test-threads=1 2>&1 | grep -oE "BASS_RESPONSE {.*}|panicked.*")
    echo "$line" | sed 's/^BASS_RESPONSE //' >> $OUT
    echo "$fx $arm $(( $(date +%s) - s ))s $(echo "$line" | head -c 60)"
  done
done
echo MATRIX-DONE

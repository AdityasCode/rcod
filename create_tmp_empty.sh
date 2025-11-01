#!/bin/bash

# pick a valid tmp dir (fall back to /tmp)
TMPDIR=${TMPDIR:-/tmp}

TMP_FILE=$(mktemp "$TMPDIR/tmp.XXXXXX")
status=$?

if [ $status -ne 0 ] || [ ! -f "$TMP_FILE" ]; then
    echo "$0: Can't create temp file in $TMPDIR, bye.."
    exit 1
fi

# write content to tmp file
cat > "$TMP_FILE" <<EOF
#!/bin/bash
$1
EOF

chmod +x "$TMP_FILE"

sbatch -c 2 "$TMP_FILE"
rm "$TMP_FILE"

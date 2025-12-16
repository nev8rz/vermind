#!/usr/bin/env bash
set -e

BRANCH="main"
DOCS_DIR="docs"
STASH_NAME="server-auto-stash"

echo "🔄 [Server] Sync start..."

# 0️⃣ 确认分支
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
  echo "❌ Not on branch '$BRANCH' (current: $CURRENT_BRANCH)"
  exit 1
fi

# 1️⃣ 保存本地修改
echo "📦 Stashing local changes..."
git stash push -u -m "$STASH_NAME" || true

# 2️⃣ 拉最新
echo "⬇️  Pulling latest changes..."
git pull --rebase origin "$BRANCH"

# 3️⃣ 恢复服务器修改
if git stash list | grep "$STASH_NAME" >/dev/null; then
  echo "📤 Restoring server changes..."
  git stash pop
fi

# 4️⃣ 禁止 docs 被修改
DOCS_CHANGED=$(git status --porcelain | awk '{print $2}' | grep "^$DOCS_DIR/" || true)
if [ -n "$DOCS_CHANGED" ]; then
  echo "❌ ERROR: Server must NOT modify '$DOCS_DIR/'"
  echo "$DOCS_CHANGED"
  exit 1
fi

# 5️⃣ 检查是否有非 docs 修改
NON_DOCS_CHANGED=$(git status --porcelain | awk '{print $2}' | grep -v "^$DOCS_DIR/" || true)
if [ -z "$NON_DOCS_CHANGED" ]; then
  echo "ℹ️  No server-side changes to commit."
  echo "✅ [Server] Sync done."
  exit 0
fi

# 6️⃣ 添加非 docs（包括新文件）
echo "🛠  Staging server-side changes..."
git add .
git reset -- "$DOCS_DIR"

# 7️⃣ 提交
git commit -m "server: update ($(date '+%Y-%m-%d %H:%M'))"

# 8️⃣ 推送
echo "⬆️  Pushing server changes..."
git push origin "$BRANCH"

echo "✅ [Server] Sync done. Repo fully synchronized."

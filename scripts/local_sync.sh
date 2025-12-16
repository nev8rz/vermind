set -e

BRANCH="main"
DOCS_DIR="docs"

echo "🔄 [Local] Sync start..."

# 1. 拉远程（保护本地修改）
echo "⬇️  Pulling latest changes..."
git pull --rebase --autostash origin "$BRANCH"

# 2. 检查是否有 docs 修改
if git status --porcelain | grep "^ M $DOCS_DIR/\\|^?? $DOCS_DIR/" >/dev/null; then
  echo "📚 Docs changed, committing..."

  git add "$DOCS_DIR"
  git commit -m "docs: update ($(date '+%Y-%m-%d %H:%M'))"
else
  echo "ℹ️  No docs changes to commit."
fi

# 3. 推送
echo "⬆️  Pushing..."
git push origin "$BRANCH"

echo "✅ [Local] Sync done. Repo is up to date."


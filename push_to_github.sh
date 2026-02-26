#!/bin/bash
################################################################################
# OpenVLA GitHub 推送指南
#
# 网络环境：透明代理（需要认证）
# 解决方案：使用 Personal Access Token 绕过代理
################################################################################

echo "=========================================="
echo "OpenVLA 推送到 GitHub"
echo "=========================================="
echo ""

cd /robot/robot-rfm/user/qiao/code/openvla

# 显示当前状态
echo "📋 当前 Git 状态："
echo "---"
git status -s
echo ""

echo "📦 待推送的提交："
echo "---"
git log --oneline -1
echo ""

echo "🔗 远程仓库："
echo "---"
git remote -v | grep myrepo
echo ""

# 方案说明
echo "=========================================="
echo "推送方案说明"
echo "=========================================="
echo ""
echo "由于网络环境存在透明代理（需要认证），"
echo "Git 无法直接推送。有以下解决方案："
echo ""
echo "方案 A: 使用 Personal Access Token（推荐）"
echo "  1. 访问: https://github.com/settings/tokens"
echo "  2. 点击 'Generate new token (classic)'"
echo "  3. 勾选 'repo' 权限"
echo "  4. 生成并复制 token"
echo "  5. 运行推送命令（见下方）"
echo ""
echo "方案 B: 配置代理认证"
echo "  如果你有代理账号密码，可以配置："
echo "  git config --global http.proxy http://user:pass@127.0.0.1:8080"
echo ""
echo "=========================================="
echo ""

# 询问用户选择
read -p "选择方案 (A/B) 或按 Ctrl+C 退出: " choice

case $choice in
    A|a)
        echo ""
        echo "📝 请输入你的 GitHub Personal Access Token:"
        echo "(输入时会隐藏显示)"
        read -s TOKEN
        echo ""
        echo ""
        echo "🚀 开始推送到 GitHub..."
        echo ""

        # 使用 token 推送
        REPO_URL="https://${TOKEN}@github.com/qiaosun22/OpenVLA_Debugged.git"

        if git push $REPO_URL main; then
            echo ""
            echo "✅ 推送成功！"
            echo ""
            echo "查看你的仓库: https://github.com/qiaosun22/OpenVLA_Debugged"
        else
            echo ""
            echo "❌ 推送失败，请检查："
            echo "  1. Token 是否正确"
            echo "  2. Token 是否有 'repo' 权限"
            echo "  3. 仓库 URL 是否正确"
            exit 1
        fi
        ;;
    B|b)
        echo ""
        read -p "请输入代理用户名: " PROXY_USER
        read -s -p "请输入代理密码: " PROXY_PASS
        echo ""

        git config --global http.proxy http://${PROXY_USER}:${PROXY_PASS}@127.0.0.1:8080
        git config --global https.proxy http://${PROXY_USER}:${PROXY_PASS}@127.0.0.1:8888

        echo "🚀 开始推送（使用代理）..."
        if git push myrepo main; then
            echo ""
            echo "✅ 推送成功！"
        else
            echo ""
            echo "❌ 推送失败，请检查代理凭据"
        fi

        # 清理代理配置
        git config --global --unset http.proxy
        git config --global --unset https.proxy
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

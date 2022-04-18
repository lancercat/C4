git checkout --orphan temp_branch
git add -A
git commit -am "publish"
git br --set-upstream temp_branch  github/main
git push -f github temp_branch
git checkout main
git branch -D temp_branch
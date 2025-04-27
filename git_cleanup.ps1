# PowerShell script for cleaning up large files from Git history

# First, update .gitignore
Write-Host "Make sure you've updated your .gitignore file before running this script!" -ForegroundColor Yellow
Write-Host "Waiting 5 seconds - press Ctrl+C to cancel..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Remove the large files from Git history
Write-Host "Removing large files from Git history..." -ForegroundColor Cyan
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch 'data/db_backup_20250427_150533/chroma.sqlite3' 'env/Lib/site-packages/torch/lib/torch_cpu.dll' 'env/Lib/site-packages/torch/lib/dnnl.lib'" --prune-empty --tag-name-filter cat -- --all

# Clean up Git's internal references
Write-Host "Cleaning up Git references..." -ForegroundColor Cyan
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "Done! Now you can push with: git push -f origin main" -ForegroundColor Green
Write-Host "WARNING: This is a force push and will overwrite the remote repository history." -ForegroundColor Red
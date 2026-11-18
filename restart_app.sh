#!/bin/bash
# Pull the latest code from origin/master and restart the server.
#
# Runtime data is never touched: history.db, gps_updates.db,
# issue_reports.log and output.log are git-ignored (see .gitignore), so
# `git reset --hard` leaves them alone. (This script used to run
# `git clean -fd`, which deletes every untracked file that isn't ignored -
# e.g. sqlite_db/version0.db - so that step is gone.)
#
# Note: `git reset --hard` still throws away code edited directly on the
# server (uncommitted changes to main.py, background_run.sh, ...). Commit
# such changes to the repo instead.

cd ~/find-my-br-train || { echo "Directory not found"; exit 1; }

# Safety copy of the user issue reports before touching git.
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p ~/deploy_backups/$TS
cp -p issue_reports.log ~/deploy_backups/$TS/ 2>/dev/null && echo "Backed up issue_reports.log to ~/deploy_backups/$TS"

PIDS=$(ps aux | grep '[g]unicorn.*main:app' | awk '{print $2}')
if [ -n "$PIDS" ]; then
    echo "Killing Gunicorn PIDs: $PIDS"
    kill $PIDS
    sleep 3
else
    echo "No running gunicorn process found."
fi

# Get the latest code and make the working tree match it exactly
git fetch origin master || { echo "git fetch failed - not restarting with unknown code"; exit 1; }
git reset --hard origin/master

# If this update stopped tracking issue_reports.log, git just deleted it -
# put the real reports back.
if [ ! -f issue_reports.log ] && [ -f ~/deploy_backups/$TS/issue_reports.log ]; then
    cp -p ~/deploy_backups/$TS/issue_reports.log . && echo "Restored issue_reports.log"
fi

bash background_run.sh

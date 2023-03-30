
#!/bin/sh

cd ~/root/hw3

# echo *.png > .gitignore
# echo ./samples/*.png > .gitignore

git add *
git commit -m "${1}"
git push -u origin master

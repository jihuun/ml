#! /bin/bash

BLOG_NAME="ml"
CUR_DIR=$(pwd)
GITBOOK_DIR="$CUR_DIR/_book"
NODEMODULS_DIR="$CUR_DIR/node_modules"
DEPLOY_DIR="$CUR_DIR/docs"
COMMIT_MSG="[skip travis] blog: $BLOG_NAME: update gitbook by TravisCI with build number $TRAVIS_BUILD_NUMBER"

echo $GITBOOK_DIR $DEPLOY_DIR

rm -r $DEPLOY_DIR/*
cp -R $GITBOOK_DIR/* $DEPLOY_DIR
rm -r $DEPLOY_DIR/docs

git clean -fx $GITBOOK_DIR

git add .
git branch
git checkout master
git commit -sm "$COMMIT_MSG"
git branch
git log -2
git push "https://${GH_TOKEN}@github.com/jihuun/$BLOG_NAME" master

#! /bin/bash

CUR_DIR=$(pwd)
GITBOOK_DIR="$CUR_DIR/_book"
NODEMODULS_DIR="$CUR_DIR/node_modules"
DEPLOY_DIR="$CUR_DIR/docs"

echo $GITBOOK_DIR $DEPLOY_DIR

rm -r $DEPLOY_DIR/*
cp -R $GITBOOK_DIR/* $DEPLOY_DIR
rm -r $DEPLOY_DIR/docs

git clean -fx $GITBOOK_DIR
git clean -fx $NODEMODULS_DIR

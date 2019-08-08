#! /bin/bash
# Need to install ImageMagick
# v 0.1.0

CLR_LIGHT_GRAY='rgb(211,211,211)'

function do_convert {
        echo $a
        $(convert  -border 1 -bordercolor $CLR_LIGHT_GRAY $1 $1)
}

cd $1
echo "Converting images in the $(pwd)"

for a in $(ls)
do
        do_convert $a
done



cd ~/root/hw3/

make clean; make 2> make-stderr.out
RunFile=./hw3

testCase=4

inFile=./samples/$testCase.png
outFile=out.png
golden_outFile=./samples/$testCase.out.png

if [ -f "$RunFile" ]; then

    echo "==================================="
    echo "=            Run hw3             ="
    echo "==================================="

    ./$RunFile $inFile $outFile

    echo "==================================="
    echo "=      Print run-stderr.out       ="
    echo "==================================="

    cat run-stderr.out


    echo "==================================="
    echo "=            Validate             ="
    echo "==================================="

    ./hw3-diff $outFile $golden_outFile
    # rm $outFile

else

    echo "==================================="
    echo "=      Print make-stderr.out      ="
    echo "==================================="

    cat make-stderr.out    

fi


echo ""
echo ""


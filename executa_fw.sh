dataset=$1;
alphaNorm=$2;
i=$3;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{print $3;}' $dataset | sort | uniq -c | wc -l`;
echo ${numClasses};
echo $i;

#./tf-idf -d treino-fold${i}.dat -t 0 > treino-norm${i}.dat
#cat treino-norm${i}.dat > treino-length${i}.dat

#cat treino-fold${i}.dat > treino-length${i}.dat

cat treino-fold${i}.dat > treino-norm${i}.dat
./lengthNorm -d treino-norm${i}.dat -t 0 -a $alphaNorm > treino-length${i}.dat

#awk -F";" '{if(NF > 3) print $0;}' treino-length${i}.dat > "treino${i}.dat"
#awk -F";" '{if(NF > 3) print $0;}' validacao-fold${i}.dat > "validacao${i}.dat"

numDocTreino=`awk -F";" 'END{print NR;}' "treino${i}.dat"`;
numDocValidacao=`awk -F";" 'END{print NR;}' "validacao${i}.dat"`;
echo "$numDocTreino - $numDocValidacao";

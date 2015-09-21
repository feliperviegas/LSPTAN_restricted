mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "validacao${i}.dat" -a $alpha -l $lambda > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
alpha=$2;
lambda=$3;
alphaNorm=$4;
i=$5;
k=$6;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{print $3;}' $dataset | sort | uniq -c | wc -l`;
echo ${numClasses};
echo $i;

./lengthNorm -d "treino-fold${i}.dat" -t 0 -a $alphaNorm > "treino-length${i}.dat"

awk -F";" '{if(NF > 3) print $0;}' "treino-length${i}.dat" > "treino${i}.dat"
awk -F";" '{if(NF > 3) print $0;}' "validacao-fold${i}.dat" > "validacao${i}.dat"

numDocTreino=`wc -l "treino${i}.dat" | awk '{print $1;}'`;
numDocValidacao=`wc -l "validacao${i}.dat" | awk '{print $1;}'`;
echo "$numDocTreino - $numDocValidacao";
mb $i

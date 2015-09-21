mb(){
i=$1;
./nb -d $k -nd $numDocTreino -nc $numClasses -nt $numAttributes -fl "treino${i}.dat" -ndV $numDocValidacao -ntV $numAttributes -ft "teste${i}.dat" -a ${alpha} -l ${lambda} > "res_nb${i}.dat" 2> "log${i}.dat";
}
dataset=$1;
k=$2;
numAttributes=`awk -f attributes.awk $dataset`;
numClasses=`awk -F";" '{print $3;}' $dataset | sort | uniq -c | wc -l`;
echo "Number of Classes ${numClasses}";
echo "Number of Attributes ${numAttributes}";

while read p;
do
	i=`echo ${p} | awk '{print $1;}'`
	alpha=`echo ${p} | awk '{print $2;}'`;
	lambda=`echo ${p} | awk '{print $3;}'`;
	alphaNorm=`echo ${p} | awk '{print $4;}'`;
	echo "Iteration $i alpha ${alpha} lambda ${lambda} alphaNorm ${alphaNorm}";

	./lengthNorm -d "treino${i}.dat" -t 0 -a ${alphaNorm} > "treino-length${i}.dat";

	awk -F";" '{if(NF > 3) print $0;}' "treino-length${i}.dat" > "treino${i}.dat";
	awk -F";" '{if(NF > 3) print $0;}' "teste${i}.dat" > tmp; mv tmp "teste${i}.dat";

	numDocTreino=`wc -l "treino${i}.dat" | awk '{print $1;}'`;
	numDocValidacao=`wc -l "teste${i}.dat" | awk '{print $1;}'`;
	echo "$numDocTreino - $numDocValidacao";
	mb $i
done <Parameters.dat

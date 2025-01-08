echo "Evaluation of the performance of the VDB and GraphDB on the same data (51 queries)";
echo;
echo "VDB";
python vdb_benchmark.py -n 51 -b
echo;
echo "Graph"
python graph_benchmark.py -n 51 -b

echo "Evaluation of the performance of the VDB and GraphDB on the same data (101 queries)";
echo;
echo "VDB";
python vdb_benchmark.py -n 101 -b
echo;
echo "Graph";
python graph_benchmark.py -n 101 -b
# Databricks notebook source
# MAGIC %md
# MAGIC ## Instructions
# MAGIC 
# MAGIC You can hit ```run all``` and go for coffee. The OPTIMIZE job takes about 10mins...

# COMMAND ----------

import re
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
localPath = "/Users/{}/demo".format(current_user)

datasetPath = "/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-*"
firstfilePath = "/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-02.csv.gz"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demonstrating Delta performance using auto-compaction and z-ordering 
# MAGIC ### Auto Compaction
# MAGIC Databricks Delta has a nifty feature to binpack and compact these small files to a configurable size (defaults to 1GB) so the downstream processing could be faster. This feature also helps in built-in data skipping as the transaction log maintains stats for the first 32 columns by default. This way we could also make the processing even faster as the data required to be scanned and files loaded will be much less. 
# MAGIC 
# MAGIC <img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/AutoCompaction.png" width=400>
# MAGIC <img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/OptimizeWrite.png" width=800>
# MAGIC 
# MAGIC ### Z Ordering
# MAGIC Z-Ordering is a technique to colocate related information in the same set of files. It maps multidimensional data to one dimension while maintaining locality. In a data context, each field we use is considered a dimension with higher resolution information that can be derived from higher dimensional data and Big Data can lead to higher orders of dimensions. 
# MAGIC Mapping things down to 1 dimension allows for concepts like range queries, which essentially cuts out scanning entire sections of data since we preserve the locality of the data in higher dimensions with the z-order curve. Z-Order with bin packing allows for sequentially efficient reads from cloud storage, decreasing total time to read data from storage into memory.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Recap on optimization technique
# MAGIC 
# MAGIC Databricks aims to simplify and automate optimization for ease of adoption and performance out of the box. Most optimization revolve around reducing the amount of data scanned.
# MAGIC 
# MAGIC Here are some optimization to look for [Link to public documentation](https://docs.databricks.com/delta/optimizations/file-mgmt.html):
# MAGIC 1. Run the latest Databricks Runtime > optimization and default parameters to improve performance
# MAGIC 2. Partition pruning
# MAGIC 3. Compaction (bin-packing) > files large enough to minimize the number of files to open
# MAGIC 4. Data skipping > collect statistics as metadata to locate files content
# MAGIC 5. Z-Ordering (multi-dimensional clustering) > reorganize/collocate data
# MAGIC 6. Tune file size > balance file size requirement: large for query speed or smaller for merge operations

# COMMAND ----------

# DBTITLE 1,Loading Dataset
spark.sql(f"CREATE DATABASE IF NOT EXISTS {dbName}")
spark.sql(f"USE {dbName}")
# get the schema from first file
df = (
  spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(firstfilePath)
)

# Load the dataset
taxi_df = (
  spark.read
    .option("header", True)
    .option("schema", df.schema)
    .format("csv")
    .load(datasetPath)
)

# COMMAND ----------

# DBTITLE 1,Transform dataset and repartition to create small files
from pyspark.sql.functions import *

# Add some parsing to date column
partitionCol = "tpep_pickup_datetime"
newColName = "PickupDate"

taxi_df = (
  taxi_df
    .withColumn(f"{newColName}_year", year(f"{partitionCol}"))
    .withColumn(f"{newColName}_month", date_format(f"{partitionCol}", "yyyyMM"))
    .withColumn(f"{newColName}_day", date_format(f"{partitionCol}", "yyyyMMdd"))
)

filterCol = f"{newColName}_day"
partitionFunction = f"{newColName}_year"

# Repartition to create small file and save to GCS
tdf = taxi_df
partitions = partitionFunction

tdf = tdf.filter("PickupDate_year = 2019").repartition(200)

tdf \
  .write \
  .partitionBy("PickupDate_year") \
  .format("delta") \
  .mode("overwrite") \
  .save(localPath)

# COMMAND ----------

# DBTITLE 1,Save the DataFrame into a table
spark.sql("DROP TABLE IF EXISTS nycmonth")
spark.sql("CREATE TABLE nycmonth USING DELTA LOCATION '{}'".format(localPath))

# COMMAND ----------

# DBTITLE 1,Check number of files read before OPTIMIZE
# MAGIC %sql
# MAGIC SELECT DISTINCT PickupDate_day FROM nycmonth

# COMMAND ----------

# DBTITLE 1,Run OPTIMIZE on table
# MAGIC %sql
# MAGIC OPTIMIZE nycmonth ZORDER BY (PickupDate_day)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Batch best practice
# MAGIC As you see above, the 200 files were compacted into <10 files (~1GB each) and *z-ordering* was done on one of the column. This process took about 10minutes and can be run as a batch job in parallel of other ETL jobs.
# MAGIC 
# MAGIC #### Streaming best practice
# MAGIC 
# MAGIC In streaming it may be worth not having autoCompact to run so not not delay micro batching. In that case a *forEachBatch* can be used to call OPTIMIZE every so many batches
# MAGIC 
# MAGIC ```
# MAGIC df.writeStream.format("delta")
# MAGIC   .foreachBatch{ (batchDF: DataFrame, batchId: Long) =>
# MAGIC     batchDF.persist()
# MAGIC     if(batchId % 10 == 0){spark.sql("optimize zOrderTarik.nycmonthevents")}
# MAGIC     if(batchId % 101 == 0){spark.sql("optimize zOrderTarik.nycmonth zorder by (PickupDate_day)")}
# MAGIC     batchDF.write.format("delta").mode("append").saveAsTable("zOrderTarik.nycmonth")
# MAGIC     batchDF.unpersist()
# MAGIC   }.outputMode("update")
# MAGIC   .start()
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Full Scan read all files
# MAGIC %sql
# MAGIC SELECT
# MAGIC   PickupDate_day,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   nycmonth VERSION AS OF 0
# MAGIC GROUP BY
# MAGIC   PickupDate_day

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   PickupDate_day,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   nycmonth VERSION AS OF 1
# MAGIC GROUP BY
# MAGIC   PickupDate_day

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparative summary - Full aggregation
# MAGIC 
# MAGIC In the 2 queries above, one was ran against the first version of the table that has 200 files, and the other one against the optmized table that was compacted. You can compare the number of files that needed to be scanned for running the query and the its impact on the query time. Below you can see the screen capture of Spark jobs i ran - but you can go for yourself in the Spark job UI in the queries above. 
# MAGIC <table>
# MAGIC     <tr>
# MAGIC       <th><h3>Full Aggregate PRE optimization</h3></th>
# MAGIC       <th><h3>Full Aggregate POST optimization</h3></th>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/DAG-FullRead.png"></td>
# MAGIC       <td><img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/DAG-FullReadPOSTOptimization.png"></td>
# MAGIC     </tr>
# MAGIC </table>

# COMMAND ----------

# DBTITLE 1,Query using 'WHERE' clause triggers data skipping - Example --before-- OPTIMIZE
# MAGIC %sql
# MAGIC SELECT
# MAGIC   PickupDate_day,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   nycmonth VERSION AS OF 0
# MAGIC WHERE
# MAGIC   PickupDate_year = 2019 AND
# MAGIC   PickupDate_day = '20190901'
# MAGIC GROUP BY
# MAGIC   PickupDate_day

# COMMAND ----------

# DBTITLE 1,Query using 'WHERE' clause triggers data skipping - Example --after-- OPTIMIZE
# MAGIC %sql
# MAGIC SELECT
# MAGIC   PickupDate_day,
# MAGIC   count(*)
# MAGIC FROM
# MAGIC   nycmonth VERSION AS OF 1
# MAGIC WHERE
# MAGIC   PickupDate_year = 2019 AND
# MAGIC   PickupDate_day = '20190901'
# MAGIC GROUP BY
# MAGIC   PickupDate_day

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparative summary - Filtered Aggregation
# MAGIC In the 2 queries above, we added a filter on the column that the table was z-ordered against. The left DAG is on the first version of the table that has 200 files, and the other one against the optmized table that was compacted. You can compare the number of files that needed to be scanned for running the query and the its impact on the query time. Below you can see the screen capture of Spark jobs i ran - but you can go for yourself in the Spark job UI in the queries above.
# MAGIC 
# MAGIC <table>
# MAGIC     <tr>
# MAGIC       <th><h3>Filter Aggregate PRE optimization</h3></th>
# MAGIC       <th><h3>Filter Aggregate POST optimization</h3></th>
# MAGIC     </tr>
# MAGIC     <tr>
# MAGIC       <td><img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/DAG-FilterPREOptimization.png"></td>
# MAGIC       <td><img src="https://github.com/tarik-missionAI/demo-databricks/raw/main/delta-perf/ressources/images/DAG-FilterPOSTOptimization.png"></td>
# MAGIC     </tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC To optimize Performance start with right compaction and right partition then check performance with off the shelve DBR (data skipping, AQE, ...) and then use z-ordering for colocating data in the file.

# COMMAND ----------



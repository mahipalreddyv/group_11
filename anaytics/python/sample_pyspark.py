from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, avg, count

# Create a Spark session
spark = SparkSession.builder \
    .appName("SamplePySparkApp") \
    .master("local[*]") \
    .getOrCreate()

# Sample data: Employee information
data = [
    (1, "Alice", "Sales", 5000),
    (2, "Bob", "Engineering", 7000),
    (3, "Charlie", "Sales", 5500),
    (4, "David", "Engineering", 7500),
    (5, "Eve", "HR", 4500),
    (6, "Frank", "Engineering", 8000),
]

columns = ["EmployeeID", "Name", "Department", "Salary"]

# Create DataFrame
df = spark.createDataFrame(data, schema=columns)

print("=" * 50)
print("Original DataFrame:")
print("=" * 50)
df.show()

# Select specific columns
print("\n" + "=" * 50)
print("Selected Columns (Name, Department):")
print("=" * 50)
df.select("Name", "Department").show()

# Filter data (salary > 5000)
print("\n" + "=" * 50)
print("Employees with Salary > 5000:")
print("=" * 50)
df.filter(col("Salary") > 5000).show()

# Group by Department and calculate statistics
print("\n" + "=" * 50)
print("Department-wise Salary Statistics:")
print("=" * 50)
df.groupBy("Department").agg(
    count("EmployeeID").alias("Count"),
    avg("Salary").alias("AvgSalary"),
    spark_sum("Salary").alias("TotalSalary")
).show()

# Order by Salary (descending)
print("\n" + "=" * 50)
print("Employees sorted by Salary (Descending):")
print("=" * 50)
df.orderBy(col("Salary").desc()).show()

# Add a new column (bonus = 10% of salary)
print("\n" + "=" * 50)
print("With Bonus Column (10% of Salary):")
print("=" * 50)
df.withColumn("Bonus", col("Salary") * 0.10).show()

# Find Second Highest Salary
print("\n" + "=" * 50)
print("Second Highest Salary Logic:")
print("=" * 50)

# Method 1: Using distinct and orderBy
second_highest = df.select("Salary").distinct().orderBy(col("Salary").desc()).limit(2).collect()
if len(second_highest) >= 2:
    second_highest_salary = second_highest[1][0]
    print(f"Second Highest Salary: {second_highest_salary}")
    print("\nEmployees with Second Highest Salary:")
    df.filter(col("Salary") == second_highest_salary).show()
else:
    print("Not enough distinct salaries to find second highest")

# Method 2: Using row_number() window function (more efficient for large datasets)
print("\n" + "=" * 50)
print("Second Highest Salary (Using Window Function):")
print("=" * 50)

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, dense_rank

# Partition by salary rank (dense_rank handles ties)
window_spec = Window.orderBy(col("Salary").desc())
df_ranked = df.withColumn("SalaryRank", dense_rank().over(window_spec))

print("Employees with Salary Rank:")
df_ranked.show()

second_highest_df = df_ranked.filter(col("SalaryRank") == 2)
print("\nEmployees with Second Highest Salary:")
second_highest_df.show()

# Stop the Spark session
spark.stop()
print("\nSpark session closed.")

Advanced Computational Statistics and Data Analysis: A Comprehensive Scenario-Based HandbookUnit I: Data Analysis using Python – Foundations, Structures, and VisualizationScenario 1.1: High-Performance Inventory Management via VectorizationScenario: A multinational logistics firm manages inventory levels for 50,000 distinct stock-keeping units (SKUs) across 20 regional distribution centers. The current legacy system uses standard Python lists to track daily stock levels and iterate through them to calculate restocking requirements. As the dataset grows, the computational latency has become unacceptable, causing delays in supply chain triggers. The Chief Data Officer requires a transition to a high-performance numerical computing architecture to handle element-wise arithmetic operations efficiently.Question: Elucidate the fundamental distinction between Python lists and NumPy vectors (arrays) within the context of large-scale data analysis. How does vectorization streamline the calculation of inventory deficits, and what are the implications for memory hierarchy?Definition and Introduction:In the domain of scientific computing with Python, the distinction between a standard list and a vector (implemented primarily via the NumPy library as ndarray) is the dividing line between general-purpose programming and high-performance data analysis. A Python list is a heterogeneous, dynamic collection of pointers to objects scattered across the heap memory, allowing for mixed data types but incurring significant overhead during iteration due to type checking and pointer dereferencing.1 Conversely, a vector is a homogeneous collection of elements stored in contiguous memory blocks, enabling the CPU to leverage Single Instruction, Multiple Data (SIMD) capabilities.Detailed Explanation:The inefficiency of Python lists for mathematical operations arises from their flexibility; the interpreter must verify the object type of every element during a loop before performing an operation. Vectorization, a core feature of NumPy, eliminates explicit Python loops by pushing the loop execution into optimized C-level code. This allows mathematical operations to be broadcast across the entire array simultaneously.For the logistics scenario, calculating restocking needs involves subtracting a "Threshold" vector from an "Inventory" vector. Using lists requires a for loop with $O(N)$ complexity in Python space. Using vectors allows this subtraction to occur at near-machine speed. Furthermore, vectors reduce memory fragmentation. While a list stores pointers (8 bytes) plus the Python object overhead (28+ bytes for an integer), a NumPy array stores raw integers (e.g., 4 or 8 bytes), resulting in a significantly smaller memory footprint, which is critical when processing 50,000 SKUs.2Python Implementation:Pythonimport numpy as np
import time

# Simulation of 1 million SKUs to demonstrate scale
n_skus = 1_000_000
inventory_list = [np.random.randint(0, 100) for _ in range(n_skus)]
threshold_list = [50 for _ in range(n_skus)]

# Approach 1: List Comprehension (Standard Python)
start_time = time.time()
deficits_list = [inv - thresh for inv, thresh in zip(inventory_list, threshold_list)]
print(f"List Execution Time: {time.time() - start_time:.4f} seconds")

# Approach 2: Vectorization (NumPy)
# The conversion to array puts data in contiguous memory
inventory_vector = np.array(inventory_list)
threshold_vector = np.array(threshold_list)

start_time = time.time()
# Element-wise subtraction without explicit loops
deficits_vector = inventory_vector - threshold_vector
print(f"Vector Execution Time: {time.time() - start_time:.4f} seconds")
Scenario 1.2: Structuring Heterogeneous Clinical Trial Data FramesScenario: A pharmaceutical research division collects multi-modal data for a Phase III clinical trial. The dataset includes Patient ID (integer), Treatment Group (categorical text), Dosage (float), and Improvement Score (float). The statisticians require a unified data structure that supports both row-wise patient retrieval and column-wise statistical operations, mimicking the utility of R's dataframes but within the Python ecosystem.Question: Explain the architecture of a Data Frame in Python. How does it act as a container for vectors, factors, and lists to manage heterogeneous data, and what is the significance of "Factors" in this context?Definition and Introduction:A Data Frame is a two-dimensional, labeled data structure capable of holding columns of potentially different data types. It serves as the primary abstraction in the pandas library, functionally equivalent to R's data.frame or a SQL table. In the hierarchy of data structures, it sits above vectors and lists, integrating them into a cohesive unit where each column is essentially a pandas Series (a specialized vector) that shares a common index.Detailed Explanation:The syllabus topic "Introduction to Vectors, Factors, Lists, Matrix and Data Frames" highlights the necessity of understanding data organization.Vectors/Series: Each column in a DataFrame is a vector. This allows for columnar operations (e.g., averaging the 'Dosage' column) to utilize NumPy's vectorized speed.Factors (Categoricals): In R, categorical data is stored as factors. In pandas, this is implemented as the category dtype. This creates a mapping where unique string labels (e.g., "Placebo", "Treatment") are mapped to integers. This reduces memory usage drastically—storing a single integer is cheaper than repeating a string 1,000 times—and speeds up operations like sorting and grouping.2Lists: DataFrames are often instantiated from dictionaries of lists, where keys become column headers and lists become the data vectors.Python Implementation:Pythonimport pandas as pd

# Raw data construction using Lists
data_payload = {
    'Patient_ID': ,
    'Group':,
    'Dosage_mg': [0.0, 50.0, 0.0, 50.0, 0.0],
    'Score': [1.2, 3.4, 1.1, 3.8, 0.9]
}

# Creating the DataFrame
df = pd.DataFrame(data_payload)

# Implementation of Factors (Categorical Data)
# This mimics R's factor() function, optimizing memory
df['Group'] = df['Group'].astype('category')

print("Data Structure Info:")
print(df.info())
print("\nFactor Levels:")
print(df['Group'].cat.categories)
Scenario 1.3: Visualizing Bivariate Economic RelationshipsScenario: An econometrician is investigating the correlation between National Advertising Expenditure and Consumer Confidence Index (CCI) over a 12-month fiscal period. To communicate these findings to stakeholders, they require a visual representation that maps the joint distribution of these two continuous variables to detect linearity or clustering.Question: What is a Scatter Plot, and how is it implemented in Python to visualize bivariate relationships? How does one customize the plot markers (pch) to differentiate data points?Definition and Introduction:A Scatter Plot is the fundamental visualization for bivariate analysis, displaying values for two variables as a set of points in Cartesian coordinates. It allows the human eye to instantly detect correlation (positive, negative, or null), clusters, and outliers. In Python, matplotlib is the foundational library for such graphics, offering granular control over every pixel. The term pch (Plot Character) is derived from R's plotting parameters, which finds its Python equivalent in the marker argument.Detailed Explanation:The syllabus specifies changing "pch from Circles to Plus Signs." In matplotlib, the default marker is 'o' (circle). To change this to a plus sign, the marker parameter is set to '+'. This customization is not merely aesthetic; distinct markers are essential when plotting multiple overlapping datasets on a single figure (e.g., urban vs. rural data) to ensure distinguishability in monochrome printing formats.Python Implementation:Pythonimport matplotlib.pyplot as plt

# Economic Data
ad_spend = 
cci_index = 

plt.figure(figsize=(10, 6))

# Syllabus Requirement: Change pch to Plus Signs, Colorful, Bigger
plt.scatter(
    ad_spend, 
    cci_index, 
    c='crimson',    # Colorful
    marker='+',     # pch equivalent
    s=150,          # Bigger size
    linewidth=2,    # Bold thickness
    label='Monthly Data'
)

# Syllabus Requirement: Add Plot Main and Axis Label Text
plt.title("Correlation: Ad Spend vs Consumer Confidence", fontsize=16, fontweight='bold')
plt.xlabel("Advertising Expenditure ($M)", fontsize=12)
plt.ylabel("Consumer Confidence Index", fontsize=12)

# Syllabus Requirement: Add text to the plot
plt.text(35, 60, "Strong Positive Trend", fontsize=12, color='blue')

plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
Scenario 1.4: Filtering and Annotating Outliers in Sales DataScenario: A retail analyst observes a massive spike in sales for "Urban" stores during a specific promotional week. To investigate, they need to filter the dataset to isolate "Urban" store records and create a visualization that explicitly labels this outlier event for the executive report.Question: How is "Filtered Data" handled in Python prior to plotting? Demonstrate the mechanism for adding specific text annotations to highlight critical data points.Definition and Introduction:Filtering is the process of subsetting a DataFrame based on Boolean conditions. Visualizing filtered data is a standard practice in Exploratory Data Analysis (EDA) to remove noise or focus on specific segments (e.g., high-value customers). "Adding text to the plot" refers to the programmatic placement of string literals at specific $(x, y)$ coordinates, used to provide context (e.g., labeling a specific outlier).6Detailed Explanation:In Pandas, filtering is achieved via boolean indexing (e.g., df == 'Urban']). Once filtered, the subset is passed to the plotting function. Annotation in matplotlib is handled by plt.text() or plt.annotate(). The latter is more powerful, allowing for arrows pointing to the data point, which is crucial for professional reporting.Python Implementation:Python# Dataset with outliers
sales_data = pd.DataFrame({
    'Store_ID': range(1, 11),
    'Type':,
    'Revenue': 
})

# Syllabus Requirement: Filtered Data (Urban only)
urban_data = sales_data == 'Urban']

plt.figure(figsize=(8, 5))
plt.scatter(urban_data, urban_data, s=100, c='green')

# Syllabus Requirement: Add text to the plot (Annotation)
# Annotating the outlier (Revenue 450)
outlier = urban_data > 400].iloc
plt.annotate(
    'Promotional Spike', 
    xy=(outlier, outlier), 
    xytext=(outlier+1, outlier+20),
    arrowprops=dict(facecolor='black', shrink=0.05),
    fontsize=10, fontweight='bold'
)

plt.title("Urban Store Revenue Analysis")
plt.xlabel("Store ID")
plt.ylabel("Revenue ($k)")
plt.show()
Scenario 1.5: Multivariate Exploration with Scatter MatricesScenario: A data scientist at a fintech startup is analyzing a dataset containing Loan Amount, Applicant Income, Credit Score, and Interest Rate. To hypothesize feature relevance for a credit risk model, they need to visualize the pairwise correlations between all these variables simultaneously.Question: What are "Multiple pairs of scatter diagrams" (Scatter Matrix), and how can they be generated in Python to visualize multivariate correlations?Definition and Introduction:A Scatter Matrix (or Pair Plot) is a grid of visualizations where the $N$ rows and $N$ columns represent the variables in the dataset. The off-diagonal cells contain scatter plots of variable $i$ vs variable $j$, while the diagonal cells typically display the univariate distribution (histogram or density plot) of variable $i$. This visualization is the gold standard for initial EDA in multivariate problems.Detailed Explanation:While matplotlib can construct this manually, the seaborn library (built on top of matplotlib) automates this with sns.pairplot. This directly addresses the syllabus requirement for "Multiple pairs of scatter diagrams." It allows the analyst to instantly spot multicollinearity (where two predictors are correlated, potentially destabilizing regression models) and separability (if coloring by a categorical variable).Python Implementation:Pythonimport seaborn as sns

# Mock Multivariate Data
fin_data = pd.DataFrame({
    'Income': ,
    'Credit_Score': ,
    'Loan_Amount': 
})

# Syllabus Requirement: Multiple pairs of scatter diagrams
# pairplot generates the N x N grid
sns.pairplot(fin_data)
plt.suptitle("Multivariate Scatter Matrix", y=1.02)
plt.show()
Scenario 1.6: Temporal Analysis with Time Series PlotsScenario: A server administrator monitors CPU temperature logs generated every minute. To predict overheating events, they need to visualize the temperature trend over the last 24 hours. The data is sequential and time-dependent.Question: How does a Time Series Plot differ from a standard scatter plot, and how is it constructed?Definition and Introduction:A Time Series Plot is a specialized line chart where the X-axis represents temporal intervals (Time, Date, Year). Unlike scatter plots, which imply no inherent order between points, time series plots assume causality and continuity between adjacent points $t$ and $t+1$. This continuity is visually represented by connecting the data points with lines.6Detailed Explanation:In Python/Pandas, handling time series involves converting string dates to datetime objects. This allows matplotlib to intelligently format the X-axis (e.g., showing "Jan", "Feb" instead of raw indices). The analysis looks for trend (long-term movement), seasonality (repeating patterns), and noise.Python Implementation:Python# Generating time-series data
dates = pd.date_range(start="2024-01-01", periods=12, freq='H')
temps = 

plt.figure(figsize=(10, 4))
# Syllabus Requirement: Time Series Plot
plt.plot(dates, temps, marker='o', linestyle='-', color='purple')
plt.title("Server Temperature Trend (12 Hours)")
plt.xlabel("Time")
plt.ylabel("Temp (°C)")
plt.grid(True)
plt.show()
Scenario 1.7: Distribution Analysis: Histograms vs. Box PlotsScenario: The HR department is auditing the salary structure of 500 employees. They need to answer two specific questions:Is the salary distribution symmetrical or skewed towards high earners?Are there specific employees earning significantly above the norm (outliers)?Question: Compare the utility of a Histogram versus a Box and Whisker Plot in this scenario.Definition and Introduction:Histogram: Displays the frequency distribution of a continuous variable by bucketing data into "bins." It provides a view of the probability density function estimate.Box and Whisker Plot: A standardized visualization of the five-number summary (Minimum, Q1, Median, Q3, Maximum). It is specifically designed to highlight the Interquartile Range (IQR) and identify outliers.Detailed Explanation:The Histogram (syllabus: "Histogram") is superior for visualizing the shape (modality, skewness). A right-skewed histogram would instantly show HR that most employees earn less, with a tail of high earners. The Box Plot (syllabus: "Box and Whisker Plot") is superior for outlier detection. Any point falling beyond $1.5 \times IQR$ from the quartiles is plotted as an individual dot, flagging it for audit.Python Implementation:Python# Generating skewed salary data
salaries = np.random.exponential(scale=50000, size=500)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram
axes.hist(salaries, bins=30, color='skyblue', edgecolor='black')
axes.set_title("Histogram: Salary Skewness")

# Box Plot
axes.boxplot(salaries, vert=False, patch_artist=True)
axes.set_title("Box Plot: Outlier Detection")

plt.show()
Scenario 1.8: Replicating R's Psych Package for Descriptive StatisticsScenario: A psychology researcher migrating from R to Python needs to generate a descriptive statistics summary for a cognitive test dataset. The syllabus explicitly references "Descriptive Statistics Using psych Package," which provides an extended summary (mean, median, trim, mad, min, max, skew, kurtosis, se). The standard df.describe() in Pandas is insufficient.Question: How can we generate a comprehensive table of Descriptive Statistics in Python that replicates the detailed output of R's psych::describe?Definition and Introduction:Standard descriptive statistics (mean, std, min, max) provide a snapshot of central tendency and dispersion. However, psychometrics often requires higher-order moments like Skewness (asymmetry) and Kurtosis (tail heaviness), and robust measures like the Median Absolute Deviation (MAD).Detailed Explanation:While the psych package is native to R, Python achieves this extended functionality by combining pandas with scipy.stats.df.describe(): Covers count, mean, std, quartiles.scipy.stats.skew / kurtosis: Adds shape metrics.scipy.stats.sem: Adds Standard Error of Mean.This manual aggregation fulfills the syllabus requirement by providing the functional equivalent in the Python environment.Python Implementation:Pythonfrom scipy.stats import skew, kurtosis, sem

# Psychometric Data
scores = pd.Series()

# Constructing the 'psych' equivalent table
desc_stats = {
    "n": scores.count(),
    "mean": scores.mean(),
    "sd": scores.std(),
    "median": scores.median(),
    "min": scores.min(),
    "max": scores.max(),
    "range": scores.max() - scores.min(),
    "skew": skew(scores),
    "kurtosis": kurtosis(scores),
    "se": sem(scores)
}

results_df = pd.DataFrame(desc_stats, index=)
print("Descriptive Statistics (Psych Equivalent):")
print(results_df.T)
Unit II: Probability and The Normal DistributionScenario 2.1: Systems Reliability and the Laws of ProbabilityScenario: A cloud computing facility uses two independent power sources, A and B. The probability of Source A failing is 0.05, and Source B failing is 0.02. The facility goes offline only if both sources fail. The manager also wants to know the probability that at least one source fails to trigger a maintenance alert.Question: Define Probability Intersection and Union. How are the Law of Addition and Multiplication used to solve this reliability problem?Definition and Introduction:Intersection ($P(A \cap B)$): The probability that outcomes A and B occur simultaneously.Union ($P(A \cup B)$): The probability that outcome A occurs, or B occurs, or both occur.Detailed Explanation:Since the failure events are independent (the failure of A does not influence B), we use the Multiplication Law for the intersection:$$P(A \cap B) = P(A) \times P(B)$$This calculates the risk of total blackout.To calculate the risk of any failure (at least one), we use the Law of Addition:$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$We subtract the intersection because simply adding $P(A) + P(B)$ would count the case where "both fail" twice.Problem Solution:Blackout (Both fail): $0.05 \times 0.02 = 0.001$ (0.1% chance).Maintenance Alert (At least one): $0.05 + 0.02 - 0.001 = 0.069$ (6.9% chance).Scenario 2.2: Conditional Probability in Fraud DetectionScenario: A bank's fraud detection system flags 1% of transactions as suspicious ($P(S) = 0.01$). If a transaction is truly fraudulent ($F$), the system flags it 99% of the time ($P(S|F) = 0.99$). However, it also flags 0.5% of legitimate transactions as suspicious (False Positive, $P(S|L) = 0.005$). If a transaction is flagged, what is the probability it is actually fraudulent?Question: How does Conditional Probability and Bayes' Theorem apply to this scenario?Definition and Introduction:Conditional Probability, denoted $P(A|B)$, is the probability of event A given that B has occurred. This is critical in inverting probabilities (getting from "Probability of Flag given Fraud" to "Probability of Fraud given Flag").Detailed Explanation:We need $P(F|S)$. Using Bayes' Theorem:$$P(F|S) = \frac{P(S|F)P(F)}{P(S)}$$Where $P(S)$ is the total probability of a flag (from both fraudulent and legitimate sources). This highlights the "Base Rate Fallacy"—even with a high detection rate, if the base rate of fraud is low, most flags might be false positives.Scenario 2.3: Project Team Selection: Permutations vs. CombinationsScenario: A software project has 12 developers. The manager needs to form a specialized "Tiger Team" of 4 members.Case A: The roles are distinct (Lead, Architect, Tester, DevOps).Case B: The roles are identical (all are General Developers).Question: Differentiate between Permutations and Combinations using Factorials in this context.Definition and Introduction:Factorial ($n!$): The product of all positive integers less than or equal to $n$.Permutation ($^nP_r$): An ordered arrangement of $r$ objects from a set of $n$. (Case A: Order matters).Combination ($^nC_r$): An unordered selection of $r$ objects from $n$. (Case B: Order doesn't matter).Detailed Explanation:Case A (Permutation): Since "Alice as Lead" is different from "Alice as Tester," we use:$$P(12, 4) = \frac{12!}{(12-4)!} = 11,880 \text{ ways.}$$Case B (Combination): Since the group {Alice, Bob, Charlie, Dave} is the same regardless of selection order, we divide by the permutations of the group itself ($4!$):$$C(12, 4) = \frac{12!}{4!(12-4)!} = 495 \text{ ways.}$$Scenario 2.4: Central Limit Theorem (CLT) in ManufacturingScenario: A semiconductor factory measures the thickness of wafers. The distribution of individual wafer thickness is highly non-normal (bimodal) due to two different machines operating. The quality engineer plans to take samples of 50 wafers and calculate the average thickness. Can they use standard statistical tests (like the Z-test) that assume normality?Question: Define the Central Limit Theorem and demonstrate it using Python simulation.Definition and Introduction:The Central Limit Theorem (CLT) states that the sampling distribution of the sample mean will approximate a Normal Distribution as the sample size increases ($n \ge 30$), regardless of the shape of the population distribution.7Detailed Explanation:This is the cornerstone of inferential statistics. It permits the use of parametric tests (Z-test, t-test) on real-world data that is rarely perfectly normal. In Python, this can be demonstrated by generating non-normal data and plotting the histogram of sample means.Python Implementation:Python# Syllabus: Central Limit Theorem Demonstration Using Python
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a Bimodal Population (Non-Normal)
pop_1 = np.random.normal(10, 2, 10000)
pop_2 = np.random.normal(30, 2, 10000)
population = np.concatenate([pop_1, pop_2])

# 2. Take 1000 samples of size 50 and compute means
sample_means =
for _ in range(1000):
    sample = np.random.choice(population, size=50)
    sample_means.append(np.mean(sample))

# 3. Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(population, bins=50, color='gray')
plt.title("Population (Non-Normal)")

plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30, color='blue', edgecolor='black')
plt.title("Sampling Distribution of Means (Normal via CLT)")
plt.show()
Scenario 2.5: The Normal Probability DistributionScenario: Standardized test scores are normally distributed with $\mu=500$ and $\sigma=100$. The exam board wants to understand the distribution's properties to set grading curves.Question: Describe the properties of the Normal Probability Distribution.Definition and Introduction:The Normal (Gaussian) distribution is a continuous, bell-shaped distribution defined by two parameters: Mean ($\mu$) and Standard Deviation ($\sigma$).Detailed Explanation:Symmetry: The curve is symmetric around the mean.Central Tendency: Mean = Median = Mode.Empirical Rule: 68% of data falls within $1\sigma$, 95% within $2\sigma$, and 99.7% within $3\sigma$.Asymptotic: The tails approach but never touch the horizontal axis (probability never hits zero).Scenario 2.6: Calculating Cumulative Risk (pnorm equivalent)Scenario: What is the probability that a student scores less than 400?Question: How do we calculate Cumulative Probability (CDF) in Python, equivalent to R's pnorm?Detailed Explanation:The R function pnorm(q, mean, sd) calculates the area under the curve to the left of $q$. The Python equivalent is scipy.stats.norm.cdf.9$$Z = \frac{400 - 500}{100} = -1.0$$The area to the left of $Z=-1$ is approximately 0.1587.Python Implementation:Pythonfrom scipy.stats import norm
# pnorm(400, 500, 100) equivalent
prob = norm.cdf(400, loc=500, scale=100)
print(f"Probability < 400: {prob:.4f}")
Scenario 2.7: Determining Percentiles (qnorm equivalent)Scenario: The university accepts the top 5% of applicants. What is the minimum score required?Question: How do we find the Quantile (Inverse CDF) in Python, equivalent to R's qnorm?Detailed Explanation:We need the score $x$ such that $P(X \le x) = 0.95$. This involves the Percent Point Function (PPF). R uses qnorm; Python uses norm.ppf. This is the inverse of the CDF.11Python Implementation:Python# qnorm(0.95, 500, 100) equivalent
cutoff = norm.ppf(0.95, loc=500, scale=100)
print(f"Top 5% Cutoff: {cutoff:.2f}")
Scenario 2.8: Probability Density for Plotting (dnorm equivalent)Scenario: To plot the curve, we need the probability density at specific points (e.g., at score 500).Question: What is the Probability Density Function (PDF) and its Python equivalent to R's dnorm?Detailed Explanation:For continuous variables, the probability of an exact value is zero. The PDF gives the height of the curve, used for plotting. R: dnorm. Python: norm.pdf.12Python Implementation:Python# dnorm(500, 500, 100) equivalent
density = norm.pdf(500, loc=500, scale=100)
print(f"Density at Mean: {density}")
Scenario 2.9: Generating Synthetic Data (rnorm equivalent)Scenario: The IT team needs 1,000 synthetic test scores to stress-test the new database.Question: How do we generate random normal variates in Python, equivalent to R's rnorm?Detailed Explanation:This uses the Random Variates (rvs) method. R: rnorm. Python: norm.rvs.9Python Implementation:Python# rnorm(5, 500, 100) equivalent
samples = norm.rvs(loc=500, scale=100, size=5)
print(f"Generated Scores: {samples}")
Unit III: Discrete Distributions and Python FunctionsScenario 3.1: Manufacturing Defects (Binomial Distribution)Scenario: A factory produces lightbulbs with a 2% defect rate ($p=0.02$). In a box of 50 bulbs ($n=50$), what is the probability of finding exactly 3 defects?Question: Define the Binomial Distribution and use Python (dbinom equivalent) to solve this.Definition and Introduction:The Binomial Distribution models the number of successes ($k$) in a fixed number of independent Bernoulli trials ($n$).$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$Here, "Success" is counter-intuitively defined as finding a defect.Detailed Explanation:R uses dbinom for the Probability Mass Function (PMF). Python uses scipy.stats.binom.pmf.13Parameters: $n=50, p=0.02, k=3$.Python Implementation:Pythonfrom scipy.stats import binom

# Probability of exactly 3 defects
prob = binom.pmf(k=3, n=50, p=0.02)
print(f"Prob of 3 defects: {prob:.4f}")
Scenario 3.2: Batch Rejection Risk (pbinom equivalent)Scenario: The quality policy states a box is rejected if it has more than 2 defects. What is the probability of a box being rejected?Question: Calculate $P(X > 2)$ using Python's equivalent to pbinom.Detailed Explanation:$$P(X > 2) = 1 - P(X \le 2)$$$P(X \le 2)$ is the Cumulative Distribution Function (CDF).R: pbinom. Python: binom.cdf.Python Implementation:Python# Prob of 0, 1, or 2 defects
prob_accept = binom.cdf(2, n=50, p=0.02)
prob_reject = 1 - prob_accept
print(f"Rejection Probability: {prob_reject:.4f}")
Scenario 3.3: Visualizing Binomial Probability (Visualize Package)Scenario: Visualizing the probability distribution is crucial for training workers. The syllabus asks for "Binomial Distribution using Visualize Package."Question: Since visualize is an R package, how do we replicate its "shaded region" functionality in Python?Detailed Explanation:The R visualize package is famous for plotting a distribution and shading the area corresponding to a p-value or probability. In Python, we achieve this manually using matplotlib. We plot the bar chart of the PMF and color the bars corresponding to the region of interest (e.g., $X > 2$).14Python Implementation:Pythonx = np.arange(0, 10)
probs = binom.pmf(x, n=50, p=0.02)

colors = ['green' if i <= 2 else 'red' for i in x]
plt.bar(x, probs, color=colors)
plt.title("Binomial Risk: Rejection Region (Red)")
plt.xlabel("Defects")
plt.show()
Scenario 3.4: Call Center Traffic (Poisson Distribution)Scenario: A help desk receives an average of 5 calls per minute ($\lambda=5$). Calculating staffing needs requires predicting probabilities of surges.Question: Define Poisson Distribution and calculate the probability of receiving exactly 8 calls (Python dpois equivalent).Definition and Introduction:The Poisson distribution models the count of events occurring in a fixed interval of time or space, assuming a constant mean rate ($\lambda$) and independence.8Formula: $P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$Python Implementation:Pythonfrom scipy.stats import poisson
# dpois(8, 5) equivalent
prob_8 = poisson.pmf(8, mu=5)
print(f"Prob of 8 calls: {prob_8:.4f}")
Scenario 3.5: Poisson Cumulative Probability (ppois equivalent)Scenario: To meet Service Level Agreements (SLA), the team must answer calls immediately. If the team can handle 8 calls/min, what is the probability they get overwhelmed (calls > 8)?Question: Calculate the tail probability using Python's ppois equivalent.Detailed Explanation:$$P(X > 8) = 1 - P(X \le 8)$$Python: poisson.cdf.Python Implementation:Python# ppois(8, 5) equivalent
prob_overwhelmed = 1 - poisson.cdf(8, mu=5)
print(f"Risk of Overload: {prob_overwhelmed:.4f}")
Unit IV: Introduction to Inferential StatisticsScenario 4.1: Medical Diagnosis: Hypothesis Testing and ErrorsScenario: A biotech firm is testing a kit for a rare disease.Null Hypothesis ($H_0$): Patient is Healthy.Alternative Hypothesis ($H_1$): Patient has Disease.Question: Explain the Types of Errors (Type I and II) and P-value in this medical context.Definition and Introduction:Type I Error ($\alpha$): False Positive. The test says "Disease" when the patient is "Healthy." This causes unnecessary panic and treatment.Type II Error ($\beta$): False Negative. The test says "Healthy" when the patient has "Disease." This is often more dangerous as the patient misses treatment.P-value: The probability of seeing the test result if the patient were truly healthy. A very low p-value (< 0.05) suggests the "Healthy" assumption is unlikely, leading to a diagnosis.Scenario 4.2: One Sample Z-Test (BSDA Package Equivalent)Scenario: A cereal manufacturer claims boxes weigh 500g ($\mu=500$). The population standard deviation is known ($\sigma=5$). A sample of 30 boxes has a mean of 498g. Is the machine underfilling?Question: Perform a One Sample Z-Test. Address the syllabus reference to "BSDA Package" (which is R-based) by providing the Python equivalent.Detailed Explanation:The Z-test applies because $\sigma$ is known and $n \ge 30$.$$Z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$$The syllabus mentions BSDA::z.test. In Python, statsmodels provides this functionality.7Python Implementation:Pythonfrom statsmodels.stats.weightstats import ztest

# Synthetic Data (Mean ~498)
data = np.random.normal(498, 5, 30)

# value=500 is the null hypothesis mean
z_stat, p_val = ztest(data, value=500, alternative='smaller')
print(f"Z-stat: {z_stat:.2f}, P-value: {p_val:.4f}")
Scenario 4.3: One Sample t-Test (Small Sample)Scenario: An automotive engineer tests a fuel additive on 10 cars ($n < 30$). The population variance is unknown. The sample shows increased MPG.Question: Why use a t-Test instead of a Z-Test? Perform this in Python.Detailed Explanation:When $\sigma$ is unknown, we estimate it using the sample standard deviation ($s$). This introduces uncertainty, requiring the Student's t-distribution (fatter tails) instead of the Normal distribution.Python: scipy.stats.ttest_1samp.17Python Implementation:Pythonfrom scipy.stats import ttest_1samp
mpg_gain = [1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0, 1.4, 0.7, 1.2]

# H0: Gain = 0
t_stat, p_val = ttest_1samp(mpg_gain, popmean=0)
print(f"T-stat: {t_stat:.2f}, P-value: {p_val:.4f}")
Scenario 4.4: Visualizing t-Test Results (Visualize Package Equivalent)Scenario: The engineer needs a graph showing the t-distribution and the rejection region to explain the result to management.Question: Visualize One Sample t-Test Results using Python (replicating the "visualize" package output).Python Implementation:Pythonfrom scipy.stats import t

df = len(mpg_gain) - 1
x = np.linspace(-4, 4, 100)
plt.plot(x, t.pdf(x, df), label='t-distribution')

# Shading p-value region
crit = t.ppf(0.95, df)
plt.fill_between(x, t.pdf(x, df), where=(x > crit), color='red', alpha=0.5)
plt.axvline(t_stat, color='black', linestyle='--', label='Observed t')
plt.title(f"One Sample t-Test Visualization (df={df})")
plt.legend()
plt.show()
Scenario 4.5: One Sample Variance Test (Chi Square Test)Scenario: A precision bolt manufacturer requires the variance of diameters to be no more than 0.01 mm². A sample of 20 bolts has a variance of 0.015.Question: Perform a One Sample Variance Test (Chi Square Test). (Syllabus mentions EnvStats, we use Python scipy).Detailed Explanation:Hypothesis: $H_0: \sigma^2 = 0.01$ vs $H_1: \sigma^2 > 0.01$.Statistic: $\chi^2 = \frac{(n-1)s^2}{\sigma_0^2}$.Degrees of freedom: $n-1$.18Python Implementation:Pythonfrom scipy.stats import chi2

n = 20
s2 = 0.015
sigma2_0 = 0.01

chi_stat = (n - 1) * s2 / sigma2_0
p_val = 1 - chi2.cdf(chi_stat, df=n-1)
print(f"Chi-Sq: {chi_stat:.2f}, P-value: {p_val:.4f}")
Scenario 4.6: Two Sample Z Test for Different MeansScenario: Comparing the average income of two large cities.City A: $n=200, \bar{x}=55k, \sigma=10k$.City B: $n=200, \bar{x}=52k, \sigma=12k$.Question: Perform a Two Sample Z Test for populations with different means.Detailed Explanation:$$Z = \frac{(\bar{x}_1 - \bar{x}_2) - 0}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}$$This formula accounts for the standard error of the difference between means.20Python Implementation:Pythonse_diff = np.sqrt((10**2/200) + (12**2/200))
z_val = (55 - 52) / se_diff
p_val = 2 * (1 - norm.cdf(abs(z_val))) # Two-tailed
print(f"Two Sample Z: {z_val:.2f}, P-value: {p_val:.4f}")
Unit V: Advanced Inference and ANOVAScenario 5.1: Two Sample t Test (Equal vs Unequal Variance)Scenario: Compare the efficacy of two fertilizers (A and B).Case 1: Levene’s test says variances are equal.Case 2: Variances are unequal (Heteroscedasticity).Question: How does the Python implementation differ for Equal Variance vs. Unequal Variance (Welch’s t-test)?Detailed Explanation:The standard Student's t-test assumes equal variance (equal_var=True). When variances differ, we use Welch's t-test (equal_var=False), which adjusts the degrees of freedom using the Welch-Satterthwaite equation to prevent Type I error inflation.Python Implementation:Pythonfrom scipy.stats import ttest_ind

group_a = 
group_b = 

# Equal Variance
t1, p1 = ttest_ind(group_a, group_b, equal_var=True)

# Unequal Variance (Welch)
t2, p2 = ttest_ind(group_a, group_b, equal_var=False)

print(f"Student's t P-value: {p1:.4f}")
print(f"Welch's t P-value: {p2:.4f}")
Scenario 5.2: Paired t Test (Pre-Post Intervention)Scenario: A fitness trainer measures the body fat percentage of 15 clients Before and After a 6-week camp.Question: Why is a Paired t Test required? Perform it in Python.Detailed Explanation:The samples are dependent (same individuals). A two-sample test would ignore the internal correlation. A paired test analyzes the differences ($d = After - Before$) against zero.Python: scipy.stats.ttest_rel.22Scenario 5.3: Two Sample Variance Test (F Test)Scenario: A bottling plant wants to buy a new machine. Machine A has a variance of 5ml. Machine B has a variance of 15ml. Is Machine B significantly less consistent?Question: Perform a Two Sample Variance Test using the F Distribution.Detailed Explanation:The F-test compares the ratio of two variances: $F = s_1^2 / s_2^2$.$H_0: \sigma_1^2 = \sigma_2^2$.If $F$ deviates significantly from 1, variances differ.Python: scipy.stats.f.23Python Implementation:PythonF_ratio = 15 / 5
df1 = 9; df2 = 9 # Assuming n=10 for both
p_val = 1 - stats.f.cdf(F_ratio, df1, df2)
print(f"F-Test P-value: {p_val:.4f}")
Scenario 5.4: Understanding ANOVA ConceptsScenario: A marketing manager compares sales across 4 regions (North, South, East, West). They propose doing 6 separate t-tests to compare every pair.Question: Explain the concept behind ANOVA and why multiple t-tests are statistically flawed (Family-wise Error Rate).Detailed Explanation:Performing multiple pairwise comparisons increases the probability of making at least one Type I error (False Positive). For 6 tests at $\alpha=0.05$, the risk is $1 - (0.95)^6 \approx 26\%$.ANOVA (Analysis of Variance) solves this by testing the global null hypothesis ($H_0: \mu_N = \mu_S = \mu_E = \mu_W$) in a single step using the F-ratio (Variance Between Groups / Variance Within Groups).Scenario 5.5: Manual ANOVA CalculationScenario:Group 1:  (Mean=2.5)Group 2:  (Mean=4.5)Grand Mean = 3.5Question: Perform the formulas and calculations for ANOVA manually (SST, SSB, SSW).Detailed Explanation:SS_Total: $\sum (x - \bar{X}_{GM})^2$.SS_Between (Factor): $n_i \sum (\bar{X}_i - \bar{X}_{GM})^2$. Measures effect of the group.SS_Within (Error): $\sum (x - \bar{X}_i)^2$. Measures random noise.F-Stat: $MS_{Between} / MS_{Within}$.Scenario 5.6: ANOVA Using PythonScenario: Run the marketing region analysis using Python.Question: Perform One-Way ANOVA using scipy.stats.Python Implementation:Pythonfrom scipy.stats import f_oneway

# Sales Data
north = 
south = 
east = 

f_stat, p_val = f_oneway(north, south, east)
print(f"ANOVA F: {f_stat:.2f}, P: {p_val:.4f}")
Scenario 5.7: Contingency Table AnalysisScenario: A survey asks 100 people their Gender and their Favorite Beverage (Coffee/Tea).Male: Coffee(30), Tea(20)Female: Coffee(10), Tea(40)Question: Construct a Contingency Table in Python and explain its use in testing independence.Detailed Explanation:A Contingency Table (Cross-tabulation) displays the frequency distribution of two categorical variables. It is the prerequisite for the Chi-Square Test of Independence (Unit IV topic, but listed in Unit V syllabus context) to check if Beverage preference depends on Gender.Python Implementation:Pythonimport pandas as pd
from scipy.stats import chi2_contingency

data = pd.DataFrame({
    'Gender': ['M']*50 + ['F']*50,
    'Bev': ['Coffee']*30 +*20 + ['Coffee']*10 +*40
})

table = pd.crosstab(data['Gender'], data)
print(table)

# Test Independence
chi2, p, dof, ex = chi2_contingency(table)
print(f"P-value: {p:.4f}")
Table 1: Summary of R vs. Python Statistical FunctionsFunctionalityR Function (Syllabus)Python Equivalent (SciPy/Statsmodels)Normal CDFpnormscipy.stats.norm.cdfNormal PDFdnormscipy.stats.norm.pdfNormal Quantileqnormscipy.stats.norm.ppfNormal Randomrnormscipy.stats.norm.rvsBinomial PMFdbinomscipy.stats.binom.pmfPoisson PMFdpoisscipy.stats.poisson.pmfDescriptive Statspsych::describepandas.describe + scipy.statsZ-TestBSDA::z.teststatsmodels.stats.weightstats.ztestVariance TestEnvStats::varTestscipy.stats.chi2 (Manual)Visualizationvisualize packagematplotlib.pyplot (Manual fill)

# UNIT 5 BDA

---

## Page 1

1. Need for Visualization
What is Data Visualization?
- Data Visualization is the process of representing data in a visual form such as charts, graphs,
maps, and diagrams so that patterns, trends, relationships, and important information can be
understood easily.
- In Big Data, the amount of data is very large and it may be difficult to understand it by looking at
rows and columns of raw data. Visualization converts this large and complex data into an easy-to-
understand visual format.
Example: Instead of analysing thousands of sales records in a table, a line chart can show how sales
changed month by month.

Why is Visualization Needed in Big Data?
Big Data contains a large volume of information, often stored across multiple servers. Looking directly at
such data in tabular form makes it difficult to identify useful information. Visualization helps users
understand this data quickly. The main needs for visualization are:
1. Easy understanding of large data
Visualization represents large datasets using graphs and charts, making the information easier to
understand than raw tables.
2. Identifying patterns and trends
Graphs help users quickly identify increasing, decreasing, or changing patterns in data. For example, a line
chart can show changes in sales over time.
3. Finding relationships
Visualization can help identify relationships between different variables. For example, a scatter plot can
be used to observe the relationship between two numerical variables.
4. Faster decision-making
Big Data analytics often produces results for managers and decision-makers. Visual representations allow
them to understand important insights quickly and support better decision-making.
5. Better communication of information
Charts, graphs, and dashboards can communicate complex information to people who may not have
technical knowledge.
6. Finding important information in one place
Visualization tools can create dashboards and reports that combine multiple insights in a single place.
These can also be shared across an organization.
7. Handling complex Big Data
Big Data may contain different types of information and may be too large to fit on a single screen.
Visualization tools help users make sense of such large and complex datasets.

Importance of Visualization in Big Data
- The importance of visualization can be summarized as:
Raw Big Data → Visualization → Patterns/Insights → Decision Making
- For example: Large sales dataset → Bar/Line Chart → Identify sales trend → Business decision
- Thus, visualization acts as a bridge between large amounts of data and meaningful insights.

---

## Page 2

Applications of Big Data Visualization
Big Data visualization can be used in many industries, such as:
- Airlines – analysing flight performance and customer feedback.
- IoT – monitoring sensor data.
- Energy – analysing energy consumption.
- Media and Entertainment – analysing audience behaviour.
- Automotive – analysing vehicle-related data.
- Sports – analysing player and team performance.
- Manufacturing – monitoring production and operational data.

2. Creating Visualization
What is Creating Visualization?
- Creating visualization is the process of converting raw or processed data into a visual
representation such as charts, graphs, maps, or dashboards so that useful information can be
understood easily.
- The type of visualization should be selected according to the purpose of analysis and the type of
data.
- The visualization can range from simple charts such as line charts, histograms and pie charts to
more complex visualizations such as scatter plots, heat maps, treemaps and 3-D graphs.

Steps for Creating a Visualization
1. Understand the Data
First, understand what type of data is available and what information it contains.
For example, data may contain:
- Numerical values
- Categories
- Time-based information
- Geographic information
- Relationships between different variables
2. Define the Purpose
Before creating a visualization, determine what you want to find or communicate from the data.
For example:
- To show changes over time → Line chart
- To compare different categories → Bar chart
- To show proportions → Pie chart
- To show frequency distribution → Histogram
- To identify relationships between variables → Scatter plot
- To show intensity or density → Heat map
3. Select an Appropriate Visualization Technique
- Choose the chart or graph that best represents the required information.
- For example, if we want to analyse monthly sales, a line chart can be used because it helps
identify trends over time. The line charts as useful for identifying trends and relationships
between variables.
4. Prepare and Process the Data
- Big Data may contain a very large number of records. Therefore, the data may need to be filtered,
aggregated, or summarized before visualization.
- For example, instead of displaying millions of individual sales records, we can calculate total sales
for each month and visualize those values.
- The aggregation and summarization techniques are used for reducing the amount of data shown
at one time.

---

## Page 3

5. Create the Visualization
After selecting the technique and preparing the data, the appropriate chart, graph, map, or dashboard is
created.
Examples include:
- Line charts
- Bar charts
- Pie charts
- Histograms
- Heat maps
- Scatter plots
- Dashboards
6. Add Interactivity if Required
For Big Data, interactive visualizations can be useful because users may need to explore different parts of
the dataset.
Common interactive features include:
- Filtering – selecting only relevant data.
- Drill-down – moving from summary information to more detailed information.
- Slicing – viewing selected categories or ranges.
- Dynamic updates – updating the visualization when new data becomes available.
7. Analyse and Interpret the Visualization
Finally, the visualization is used to identify:
- Patterns
- Trends
- Relationships
- Differences
- Anomalies
- Important insights
The purpose is not simply to create a beautiful graph, but to understand the information represented
by the data.

Example : Suppose a company has a large dataset containing monthly sales of different products.
1. Collect the sales data.
2. Group the data by month and product.
3. Calculate total sales.
4. Select a line chart to observe sales trends.
5. Add filters for different products.
6. Analyse which products have increasing or decreasing sales.
Thus, the visualization makes the large sales dataset easier to analyse.

Tools Used for Creating Visualizations
The PPT mentions several tools and platforms:
- Tableau – interactive dashboards and visualization.
- Power BI – interactive reporting and integration with different data sources.
- Apache Superset – open-source data visualization and exploration.
- Kibana – visualization of Elasticsearch data.
- Matplotlib and Seaborn – Python visualization libraries.
- Plotly – interactive plots and dashboards.
- D3.js – highly customizable web-based visualizations.
- Three.js – 3-D visualizations.

---

## Page 4

3. Existing Visualization Techniques
Big Data can be represented using different visualization techniques depending on the type of data and
purpose of analysis. Common techniques include line charts, histograms, bar charts, pie charts, heat maps,
scatter plots, and box plots.

3.1 Line Chart
A line chart represents data points using points connected by lines. It is mainly used to show changes or
trends over time and relationships between two variables.
Example: Monthly sales of a company can be represented using a line chart to see whether sales are
increasing or decreasing.
Used for:
- Time-based data
- Identifying trends
- Comparing changes over time
3.2 Histogram
- A histogram represents the frequency distribution of numerical data. It divides data into different
ranges or intervals and shows how many data points fall within each range.
- It helps understand the distribution of data. The distribution can be symmetric, right-skewed, or
left-skewed.
Example: A histogram can show how many students fall into different age ranges.
Used for:
- Frequency distribution
- Understanding data distribution
- Identifying the concentration of values
3.3 Bar Chart
A bar chart represents categorical data using rectangular bars. The length or height of each bar represents
the value or quantity of that category. Bars can be vertical or horizontal.
Example: A bar chart can show the number of people who prefer different types of movies.
Used for:
- Comparing categories
- Comparing quantities
- Showing rankings or differences between groups
3.4 Pie Chart
A pie chart represents data as a circle divided into different slices. Each slice represents a proportion of
the total data, and its size depends on the relative value of that category.
Example: A company's total expenditure can be divided into categories such as salaries, marketing,
infrastructure, and transportation.
Used for:
- Showing proportions
- Showing percentage distribution
- Comparing parts of a whole
3.5 Heat Map
A heat map uses colours to represent values or ranges of values in a two-dimensional representation.
Different colours or colour intensities help users quickly identify high and low values.
Example: A heat map can show temperature variations across different cities and months.
Used for:
- Showing intensity
- Finding high and low values
- Identifying patterns in large datasets
- Showing data density

---

## Page 5

3.6 Scatter Plot
- A scatter plot uses individual dots or points to represent numerical values. The position of each
point on the X-axis and Y-axis represents the values of two variables.
- It is useful for observing whether there is a relationship or pattern between two numerical
variables.
Example: A scatter plot can show the relationship between a person's height and weight.
Used for:
- Relationship between two variables
- Identifying patterns
- Identifying clusters or unusual observations
3.7 Box Plot
A box plot is used to represent the distribution of numerical data using important statistical values. It
provides a compact way to understand the spread and central tendency of the data.
It can help in understanding values such as:
- Median
- Lower and upper quartiles
- Minimum and maximum values
- Spread of data
- Possible outliers
The PPT includes box plot as one of the visualization types and also relates it to the relationship between
median, mean and mode.
Example: A box plot can be used to compare the distribution of salaries among different departments.

Relationship between Mean, Median and Mode
Mean, Median and Mode are three important measures used to describe the central tendency of a
dataset.
- Mean: The average value of all observations.
- Median: The middle value when the data is arranged in ascending or descending order.
- Mode: The value that occurs most frequently.
Their relationship depends on the shape or distribution of the data.
1. Symmetrical Distribution
When the data is symmetrically distributed: Mean = Median = Mode
The three measures are located approximately at the centre of the distribution.
2. Positively Skewed Distribution
When the distribution is right-skewed, the larger values pull the mean towards the right.
Therefore: Mode < Median < Mean
3. Negatively Skewed Distribution
When the distribution is left-skewed, the smaller values pull the mean towards the left.
Therefore: Mean < Median < Mode

Distribution
Relationship
Symmetrical
Mean = Median = Mode
Right/Positive Skew
Mode < Median < Mean
Left/Negative Skew
Mean < Median < Mode
Exam point: This relationship helps us understand the shape and skewness of a dataset.

### Table 1

| Distribution | Relationship |
| --- | --- |
| Symmetrical | Mean = Median = Mode |
| Right/Positive Skew | Mode < Median < Mean |
| Left/Negative Skew | Mean < Median < Mode |

---

## Page 6

4. Big Data Visualization Challenges
Big Data visualization is difficult because Big Data contains very large volumes of complex and
continuously generated data. Traditional visualization methods may not be sufficient to handle such
datasets effectively.
4.1 Data Volume
Challenge: Big Data contains a huge number of records. Visualizing all the data at once can cause
performance problems and make it difficult to identify meaningful information.
Technique/Solution: Use:
- Sampling – represent the dataset using a suitable subset.
- Aggregation – summarize data using operations such as sum or average.
- Filtering – display only relevant data.
4.2 Data Variety
Challenge: Big Data can come in different formats such as structured, semi-structured, and unstructured
data and may come from multiple sources. Combining and visualizing these different types of data can be
difficult.
Technique/Solution: Use data preprocessing, normalization, and integration to make data from different
sources suitable for visualization.
4.3 Data Velocity
Challenge: Big Data can be generated at a very high speed. In real-time applications, data must be
processed and visualized quickly.
Technique/Solution: Use stream processing and real-time analytics platforms to process and visualize
continuously generated data.
Example: Apache Kafka and Apache Flink can be used for real-time data processing.
4.4 Data Complexity
Challenge: Big Data may contain multiple dimensions, variables, and relationships, making it difficult to
represent everything clearly in a single visualization.
Technique/Solution:
Use:
- Dimensionality reduction, such as PCA.
- Multi-dimensional charts.
- Interactive dashboards.
These techniques simplify complex datasets while retaining important information.
4.5 Perceptual Scalability
Challenge: Human users cannot easily understand all relevant information when a visualization contains
a very large amount of data. Even the size of the screen can limit how much information can be displayed
at once.
Solution:
Use techniques such as:
- Data aggregation
- Filtering
- Sampling
- Data reduction
This reduces visual overload and makes the visualization easier to understand.
4.6 Real-Time Scalability
Challenge: Users often expect information to be available in real time, but processing and visualizing
large datasets takes time. Therefore, providing real-time visualization for Big Data can be difficult.
Solution: Use real-time data streaming and processing technologies to handle continuously generated
data.
4.7 Interactive Scalability
Challenge: Interactive visualization allows users to explore datasets, but as the size of the dataset
increases, the visualization may take longer to respond. In extreme cases, the system may freeze or crash.

---

## Page 7

Solution: Use scalable visualization frameworks, filtering, aggregation, and efficient data processing to
improve response time.
4.8 Data Quality
Challenge: Incomplete, incorrect, or inaccurate data can produce misleading visualizations and incorrect
conclusions.
Solution:
Perform:
- Data cleaning
- Data validation
- Data preprocessing
Visualization can also be used to identify anomalies or unusual data values.
4.9 User Experience
Challenge: A visualization should be understandable to users with different levels of technical
knowledge. A complicated or poorly designed visualization can make interpretation difficult.
Solution:
Follow user-centered design principles and create visualizations that are:
- Clear
- Simple
- Concise
- Easy to interpret
Context and guidance can also be provided to help users understand the information.
4.10 Scalability
Challenge: Visualization tools must continue to work efficiently even when the size of the dataset
increases significantly.
Solution:
Use:
- Scalable visualization frameworks
- Distributed computing resources
- Tools such as D3.js and Apache Superset
These can support visualization of large-scale datasets.

5. Techniques for Visualizing Big Data
- Big Data contains a very large amount of information, so traditional charts alone may not always
be sufficient.
- Big Data visualization techniques help reduce complexity, represent large datasets effectively, and
make important patterns and insights easier to identify.
- The PPT lists several techniques for effective Big Data visualization.
5.1 Aggregation and Summarization
- Aggregation and summarization reduce the amount of information displayed by combining large
numbers of data values into meaningful summaries.
- For example, instead of displaying the sales of every transaction, we can show the total or average
sales for each month.
Common operations include:
- Grouping data
- Calculating averages
- Calculating sums
- Summarizing data
This makes large datasets easier to visualize and understand.

---

## Page 8

5.2 Heat Maps and Geographic Maps
1. Heat Maps - represents values using different colours or colour intensities. It is useful for showing data
density or intensity.
Example: A heat map can show areas with high and low website activity.
2. Geographic Maps - used when the data is related to locations or geographical areas.
Example: A company can use a map to visualize the number of customers in different cities.

5.3 Interactive Dashboards
An interactive dashboard combines different visualizations into a single interface and allows users to
explore the data.
Users can:
- Apply filters
- Select categories
- Explore different parts of the data
- View updated information
Tools such as Tableau, Power BI, and Looker can be used to create interactive dashboards.
Example: A business dashboard can display sales, profit, customer count, and regional performance
together.

5.4 Hierarchical and Network Visualizations
These techniques are useful for representing relationships, structures, and connections within data.
1. Hierarchical Visualization - It represents data arranged in different levels or categories.
Examples:
- Treemap
- Sunburst diagram
A treemap represents hierarchical data using nested rectangles.
Example: A company's organizational structure can be represented hierarchically.
2. Network Visualization - A network graph represents relationships or connections between
different entities.
Example: A social network can represent people as nodes and their connections as links.

5.5 Advanced Graphing Techniques
Advanced graphing techniques are used when the dataset is complex and simple charts are not sufficient.
The PPT mentions:
- 3D visualizations
- Parallel coordinates
- Multi-dimensional scaling
These techniques help represent complex or multi-dimensional datasets visually.
Example: A 3D visualization can represent three dimensions of data simultaneously.

5.6 Real-Time Data Streaming
- In some applications, data is generated continuously and needs to be visualized as it arrives.
- Real-time data streaming allows continuously generated data to be processed and visualized with
minimum delay.
Technologies mentioned in the PPT include:
- Apache Kafka
- Apache Flink
These technologies support real-time data streaming and processing.
Example: An IoT system can continuously receive sensor data and display the current readings on a
dashboard.

---

## Page 9

5.7 Custom Visualization Solutions
Sometimes standard visualization tools may not provide exactly what is required. In such cases, custom
visualizations can be developed according to specific requirements.
The PPT mentions:
- D3.js – used to create highly customizable and interactive web-based visualizations.
- Three.js – used for 3D visualizations.
Example:
A company may develop a customized interactive web visualization for a specialized dataset instead of
using a standard chart.

6. Key Techniques for Visualizing Big Data
- The key techniques that make Big Data visualization more manageable, interactive, and useful for
analysis.
- These techniques include data aggregation and filtering, interactive visualization, advanced chart
types, geographic visualization, temporal analysis, dimensionality reduction, and custom
visualization.

6.1 Data Aggregation and Filtering
Data Aggregation - Data aggregation means combining a large number of data values into a smaller and
meaningful summary.
- For example, instead of displaying sales for every individual transaction, we can calculate the
total sales for each month.
Common aggregation operations include:
- Sum
- Average
- Count
- Minimum
- Maximum
Aggregation helps reduce the amount of data that needs to be displayed and improves visualization
performance.
Data Filtering - Data filtering means displaying only the relevant portion of a dataset.
- For example, from a dataset containing sales from all countries, we can filter and display only
sales from India.
Filtering helps:
- Reduce information overload.
- Focus on relevant data.
- Make visualization easier to understand.

6.2 Interactive Visualizations
Interactive visualization allows users to explore the data instead of only viewing a static chart.
The PPT mentions three important interactive features:
a. Drill-Down
Drill-down allows users to move from a high-level summary to more detailed information.
Example: Country → State → City → Individual store
So, the user can start with overall sales and click further to see detailed sales for a particular region.
b. Filtering and Slicing
Users can select a particular category, range, or subset of data to view.
Example: A sales dashboard can be filtered by:
- Year
- Product

---

## Page 10

- Region
This allows users to focus only on the information they need.
c. Dynamic Updates - Dynamic updates allow the visualization to change when new data becomes
available.
This is especially useful for real-time or continuously changing data.

6.3 Advanced Chart Types
Advanced charts are useful for representing complex Big Data that may not be easily represented using
basic charts.
a. Heat Map - A heat map uses colour gradients to represent the density or intensity of data.
Example: Showing areas with high and low customer activity.
b. Tree Map - A treemap represents hierarchical data using nested rectangles.
The size of the rectangles can represent the value of a particular category.
Example: Representing a company's sales by country → state → city.
c. Network Graph
A network graph represents relationships and connections between different entities.
For example, in a social network:
- People can be represented as nodes.
- Connections between people can be represented as links.
d. Bubble Chart
A bubble chart represents three dimensions of data using position and size.
For example:
- X-axis → Sales
- Y-axis → Profit
- Bubble size → Number of customers
Thus, a bubble chart can represent multiple dimensions in one visualization.

6.4 Geographic and Spatial Visualization
Geographic and spatial visualization is used when data is associated with locations or geographical areas.
1. Maps - Maps can be used to represent location-based data and spatial patterns.
Example: A company can display the distribution of customers across different cities on a map.
2. Geospatial Analysis - Geospatial analysis combines data with Geographic Information Systems
(GIS) to perform more advanced analysis of geographical information.

6.5 Temporal Analysis - Temporal analysis means analysing how data changes over time.
a. Time-Series Charts - Time-series charts show changes in data over a period of time.
Common examples include:
- Line charts
- Area charts
Example: Monthly sales from January to December can be shown using a time-series chart.
b. Animations - Animated charts can be used to show changes and trends dynamically over time.
For example, population changes across different years can be represented using animation.

6.6 Dimensionality Reduction
- Big Data can contain a large number of variables or dimensions. Visualizing all these dimensions
directly can be difficult.
- Dimensionality reduction reduces the number of dimensions while trying to retain important
information from the original data.
The PPT mentions two techniques:

---

## Page 11

a. Principal Component Analysis (PCA) - PCA reduces the number of dimensions in a dataset while
preserving as much of its important variation as possible. It is useful when a dataset contains many
variables and we want to represent the data using fewer dimensions.
Example: A dataset with 10 variables can potentially be transformed into a smaller number of principal
components for easier visualization.
b. t-SNE
t-Distributed Stochastic Neighbor Embedding (t-SNE) is used to visualize high-dimensional data in lower
dimensions.
It is often useful for identifying:
- Clusters
- Patterns
- Groups of similar data points

6.7 Custom Visualizations - Sometimes standard charts and visualization tools are not sufficient for a
specific requirement. In such cases, custom visualizations can be created.
- D3.js - D3.js is a JavaScript library used to create highly customizable and interactive web-based
visualizations.
- Three.js - Three.js can be used to create 3D visualizations, allowing complex data to be
represented in an immersive visual form.

7. Visual Analysis of Big Data
7.1 Meaning of Visual Analysis
- Visual Analysis of Big Data is the process of using charts, graphs, maps, dashboards, and
interactive visualizations to explore, understand, and interpret large and complex datasets.
- Instead of examining thousands or millions of individual records, visual analysis represents the
data graphically so that patterns, trends, relationships, and unusual values can be identified more
easily.
Example: A company can use a dashboard to visualize sales data and identify which products have high
sales, which regions are performing poorly, and how sales change over time.

7.2 Main Purposes of Visual Analysis
Visual analysis mainly helps in three types of analysis:
1. Exploratory Analysis
- Exploratory analysis is used to explore the data and discover patterns, trends, relationships, and
anomalies.
- It is generally performed before making final conclusions.
Example: A retailer uses a scatter plot to find whether advertising expenditure is related to sales.
Main purpose: To find “What is happening in the data?”

2. Explanatory Analysis
- Explanatory analysis is used to present and communicate the important findings from data to
other people such as managers, customers, or decision-makers.
- The visualization should be clear and easy to understand.
Example: A bar chart showing the yearly revenue of a company can be presented to management.
Main purpose: To explain “What did we find?”

3. Diagnostic Analysis
- Diagnostic analysis is used to identify the reasons or causes behind a particular pattern, problem,
or result.
- It goes deeper into the data to understand why something happened.

---

## Page 12

Example: If sales suddenly decrease, diagnostic analysis can help determine whether the reason was low
demand, increased price, supply problems, or poor customer service.
Main purpose: To understand “Why did it happen?”

7.3 Principles of Visual Analysis
Effective visual analysis should follow four important principles.
1. Clarity
The visualization should be simple, clear, and easy to understand.
- Avoid unnecessary elements.
- Use proper labels and titles.
- Choose suitable charts.
Example: A simple bar chart is better than a complicated graph when comparing product sales.
2. Relevance
- The visualization should show information that is relevant to the objective of the analysis.
- Unnecessary data should be avoided.
- Example: If the objective is to compare sales by region, the visualization should focus on regional
sales.
3. Accuracy
- The visualization must represent the data correctly and without distortion.
- Incorrect scales, misleading charts, or poor-quality data can result in wrong conclusions.
- Example: Using an inappropriate scale in a bar chart may make a small difference look very
large.
4. Interactivity
Interactive visualization allows users to explore the data dynamically.
Users can:
- Filter data
- Zoom in or out
- Drill down into details
- Select specific categories
- Change time periods
Example: In a sales dashboard, a manager can select a particular year and region to view only that data.

7.4 Data Visualization Types Used in Visual Analysis
Different types of charts are used depending on the type of information that needs to be analyzed.
1. Line Chart - A line chart shows how values change over time.
Used for:
- Trends
- Time-series data
- Growth or decline
Example: Monthly sales of a company.

2. Bar Chart - A bar chart compares values between different categories.
Used for:
- Category comparison
- Ranking
- Frequency comparison
Example: Sales of different products.

3. Pie Chart - A pie chart represents parts of a whole using different slices.
Used for:

---

## Page 13

- Proportions
- Percentages
- Composition
Example: Percentage of customers using different payment methods.

4. Histogram - A histogram represents the frequency distribution of numerical data by grouping values
into ranges.
Used for:
- Distribution
- Frequency
- Understanding data shape
Example: Distribution of students' marks.

5. Scatter Plot - A scatter plot represents the relationship between two numerical variables.
Used for:
- Correlation
- Relationships
- Outlier detection
Example: Relationship between advertising expenditure and sales.

6. Heat Map - A heat map uses colors to represent the magnitude of values.
Used for:
- Large datasets
- Pattern identification
- Comparing values across two dimensions
Example: Website activity by day and hour.

7. Box Plot - A box plot summarizes the distribution of numerical data using values such as:
- Minimum
- First quartile
- Median
- Third quartile
- Maximum
It is also useful for identifying outliers.
Example: Comparing salary distributions of different departments.

7.5 Advanced Visualization Techniques
Big Data often contains many dimensions and complex relationships. Advanced techniques help represent
such data effectively.
1. 3D Visualization
- 3D visualization represents data using three dimensions: X-axis, Y-axis, and Z-axis.
- It is useful when multiple numerical dimensions need to be viewed together.
Example: Visualizing sales according to product, region, and time.
Tools such as Three.js and WebGL can be used for 3D visualization.
2. Time-Series Visualization - represents how data changes over a period of time.
Common methods include:
- Line charts
- Animated charts
- Sliding time windows
Example: Monitoring temperature every hour for several days.

---

## Page 14

3. Hierarchical Visualization - represents data arranged in levels or parent-child relationships.
Common techniques include:
- Treemap
- Sunburst chart
- Dendrogram
Example: Representing an organization's structure from company → departments → teams → employees.

4. Dendrogram
- A dendrogram is a tree-like diagram used to represent hierarchical relationships or clusters.
- It is commonly used in hierarchical clustering.
Example: Grouping customers based on similar purchasing behavior.

7.6 Interactive Exploration
Interactive exploration allows users to directly interact with visualized data and examine different parts
of a large dataset.
Important operations include:
1. Filtering - Shows only the required subset of data.
Example: Display sales only for Mumbai.
2. Drill-down - Moves from summarized information to more detailed information.
Example: Country → State → City → Store.
3. Zooming - Allows users to focus on a particular part of the visualization.
4. Panning - Allows users to move around a large visualization to examine different sections.
These features make large datasets easier to explore and understand.

7.7 Big Data Platforms for Visual Analysis
Big Data platforms help process and analyze large datasets before or along with visualization.
1. Apache Hadoop - Apache Hadoop is a framework used for storing and processing very large
datasets across distributed systems. It can be combined with visualization tools to analyze large
amounts of data.
2. Apache Spark - Apache Spark is a distributed data processing framework used for fast
processing and analysis of large datasets. It can process data and integrate with different
visualization tools for analysis.

CASE STUDIES
Case Study 1: Smartphone Reviews Analysis
A customer service team has a large dataset of smartphone reviews, ratings, and timestamps. The
objective is to understand customer sentiment, identify common issues, and study how opinions change
over time.
Suitable visualizations:
1. Pie Chart / Stacked Bar Chart – Sentiment Distribution
Used to show the proportion of:
- Positive reviews
- Neutral reviews
- Negative reviews
This helps the company quickly understand the overall customer sentiment.
2. Word Cloud – Common Issues
A word cloud can show frequently occurring words or topics in reviews. It can be created separately for
positive and negative reviews to identify common customer concerns.
3. Line Chart – Sentiment Over Time
A line chart can show how positive, neutral, and negative sentiment changes over time.

### Table 1

|  | CASE STUDIES |  |
| --- | --- | --- |
| Case Study 1: Smartphone Reviews Analysis |  |  |

---

## Page 15

Example: If negative reviews suddenly increase after a product update, the company can investigate the
reason.
Exam Point:
Pie/Stacked Bar → overall sentiment
Word Cloud → common issues/topics
Line Chart → sentiment trend over time

Case Study 2: News Media Sentiment Analysis
A news organization wants to analyze how sentiment related to different political figures changes over
time and how different media sources or platforms differ in their coverage.
Suitable visualizations:
1. Multiple Line Charts – Sentiment Over Time
A separate line can represent each figure, allowing sentiment trends to be observed over different time
periods.
2. Stacked Area Chart / Heat Map – Comparison
These can be used to compare sentiment patterns across multiple figures simultaneously.
3. Bar Chart / Heat Map – Media Source Comparison
These can be used to compare sentiment or coverage patterns across different media sources or
platforms.
Exam Point:
Line Chart → sentiment over time
Stacked Area/Heat Map → compare multiple figures
Bar/Heat Map → compare media sources

Case Study 3: Retailer Social Media Sentiment
A retailer wants to analyze customer sentiment across Twitter, Instagram, and Facebook, understand
product sentiment over time, and study the relationship between engagement and sentiment.
Suitable visualizations:
1. Line Chart – Platform Trends
Separate lines can represent Twitter, Instagram, and Facebook to compare sentiment trends over time.
2. Multi-Line Chart / Stacked Area Chart – Product Sentiment
These charts can show how sentiment for different products changes over time.
3. Scatter Plot – Engagement vs Sentiment
A scatter plot can show whether there is a relationship between social media engagement and customer
sentiment.
For example, engagement can be plotted on one axis and sentiment score on the other.
Exam Point:
Line Chart → platform trends
Multi-Line/Stacked Area → product sentiment
Scatter Plot → engagement vs sentiment

Case Study 4: Airline Customer Sentiment
An airline wants to analyze customer sentiment regarding different factors such as service, timeliness,
cleanliness, and comfort and compare its performance with competitors.
Suitable visualizations:
1. Radar Chart – Sentiment by Factors
A radar chart can compare customer sentiment across different service factors such as:
- Service quality
- Timeliness
- Cleanliness
- Comfort

---

## Page 16

It gives an overall visual profile of different aspects of the airline experience.
2. Line Chart – Trends Over Time
A line chart can show how sentiment for each factor changes over time.
3. Grouped/Clustered Bar Chart – Competitor Comparison
A grouped bar chart can compare the airline's performance with competitors or industry averages across
different factors.
Exam Point:
Radar Chart → compare service factors
Line Chart → changes over time
Grouped Bar Chart → compare airlines/competitors
.

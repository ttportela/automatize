### Multiple Aspect Trajectory - File Format (v1.0)


![Multiple Aspect Trajectory!](/assets/mat.svg "Multiple Aspect Trajectory")


The .MAT file format is a form of representing the complex data of Multiple Aspect Trajectories. Following is an example file: 


```
# comment description lines
@problemName DatasetName
@missing True
@aspects 6
@aspectNames Aspect1,Aspect2,Aspect3
@trajectoryAspectNames PersistentAspect1,PersistentAspect2
@label label
@aspectDescriptor Aspect1:nominal,Aspect2:numeric,Aspect3:numeric,PersistentAspect1:timestamp,PersistentAspect2:nominal,tid:numeric,label:nominal
@data
@trajectory 
4,user1
@trajectoryAspects
3909101102931293,Student
@trajectoryPoints
Class Room A,1,67.0
Chemistry Lab,1,86.0
Class Room B,3,99.0
@trajectory 
.
.
.
```

#### 1. Comment lines

Comments are allowed in the begining of the file preceeded by the '#' character, and only before any metadata or data. Used for a textual description of the dataset or other information.

```Bash
# comment line 1
# comment line 2
```

#### 2. Dataset Attributes

Dataset attributes are tags for the dataset representing pairs of key and value. The key is preceeded by the '@' character, following the value separated by one space:

```
@problemName DatasetName
@missing True
@aspects 6
```

Standard dataset attributes are:
* `@problemName`: the dataset name (the value must follow a `TittleCase` variable naming rule);
* `@missing`: boolean attribute (`True` or `False`) indicating is the dataset has missing values, where the missing velues are represented by the '?' character;
* `@aspects`: number of aspect columns in the dataset.

New dataset attributes may be created as needed and are **not required**.

#### 3. Dataset Metadata

Dataset metadata are **required** for the dataset represented data of the trajectories. The key is preceeded by the '@' character, following a space char and the subsequent metadata values are separated by commas.

```
@aspectNames Aspect1,Aspect2,Aspect3
@trajectoryAspectNames PersistentAspect1,PersistentAspect2
@label label
@aspectDescriptor Aspect1:nominal,Aspect2:numeric,Aspect3:numeric,PersistentAspect1:timestamp,PersistentAspect2:nominal,tid:numeric,label:nominal
```

* `@aspectNames`: list of names for the trajectory point attributes, i.e., the attribute names of each trajectory point;
* `@trajectoryAspectNames`: list of names for the trajectory persistent attributes, i.e., the attributes that are the same throughout the entire trajectory;
* `@aspectDescriptor`: the data types of each attribute comma separated `key:value` pairs. This includes the `label` column that represents the **moving object** (not required) and the `tid` column (required). The `tid` represents the Trajectory ID;
* `@label` (optional): the trajectory persistent attributes that can be used as label for classification tasks;
* `@aspectComparator` (optional): the type of distance measure used by default for each attribute (comma separated `key:value` pairs).


The Attributes and Metadata section of the file ends with the line of `@data` tag, which should be in one line alone. After this line the following lines represents each trajectory data.

#### 4. Trajectory Data

The trajectory data has three sections, each one marked by its tag followed by the data in a new line:

##### 4.1. Tag `@trajectory`: 

The trajectory tag preceeds the trajectory ID (`tid`) and can be followed by the **Moving Object** value (separated by a comma).

```
@trajectory 
4,user1
```

##### 4.2. Tag `@trajectoryAspects`: 

This tag marks the comma separated values of the trajectory persistent aspect values (should be just one line of data). Each value are in the order presented by `@trajectoryAspectNames` tag.

```
@trajectoryAspects
3909101102931293,Student
```

##### 4.3. Tag `@trajectoryPoints`: 

This tag marks the trajectory points data. Each following line represents one point of the trajectory with comma separated values for each aspect in the same order as presented by `@aspectNames` tag. 

```
@trajectoryPoints
Class Room A,1,67.0
Chemistry Lab,1,86.0
Class Room B,3,99.0
```


---



#### 5. Complementary Definitions

Following are presented other information about the MAT file format.

##### 5.1. Aspect Types

Each aspect descriptor (`@aspectDescriptor`) can be of one of the following types:

* `nominal` or `text`: textual value;
* `numeric`: integer or floating point numerical value;
* `space2d`: *x* and *y* positions of a point in a 2-dimensional spatial area, separated by space as `"12.3 32.1"`;
* `space3d`: *x*, *y*, and *z* positions of a point in a 3-dimensional spatial area, separated by spaces as `"12.3 23.1 32.1"`;
* `timestamp`: timestamp of a date type;
* `time`: numeric value representing the time of a date as the hour or the minute of the day;
* `datetime`: textual representation of a date in the format: `"yyyy-MM-dd HH:mm:ss"`;
* `localdate`: Serialized form of a Java `LocalDate`. A date without a time-zone in the ISO-8601 calendar system, such as 2007-12-03;
* `localtime`: Serialized form of a Java `LocalTime`. A time without a time-zone in the ISO-8601 calendar system, such as 10:15:30.

##### 5.1. Aspect Comparators (optional)

Each aspect comparator (`@aspectComparator`) can be of one of the following types of distance functions, names are not case sensitive. By default they are implemented in **[HiPerMovelets](/method/HIPERMovelets)** method:

```
@aspectComparator Aspect1:equals,Aspect2:difference,...
```


* Nominal types:
    * `equals`: simple text equality;
    * `equalsignorecase`: simple text equality without case letters sensitivity;
    * `weekday`: zero distance if both values are either weekday or are weekend;
    * `lcs`: Longest Commom Subsequence distance for text;
    * `editlcs`: Edit distance text;
    * `wordlcs`: Longest Commom Subsequence of words in the text;
    * `wordeditlcs`: Edit distance of words in the text;


* Numeric types:
    * `difference`: simple numerical difference (normalized);
    * `diffnotneg`: simple non negative numerical difference;
    * `equals`: simple numeric equality;
    * `proportion`: proportion relation: `abs(n1 - n2) / abs(n1 + n2)`;

 
* Space2D types:
    * `euclidean`: spatial Euclidean distance;
    * `manhattan`: spatial Manhattan distance;


* Space3D types:
    * `euclidean`: spatial Euclidean distance of 3D space;


* Time types:
    * `difference`: simple numerical difference;


* DateTime types:
    * `difference`: simple numerical difference of time in seconds;


* LocalDate types:
    * `difference`: simple numerical difference of time in seconds;
    * `diffdaysofweek`: simple difference of days between two weekdays;
    * `equaldayofweek`: equality if same day of week;
    * `isweekend`: zero distance if day is a weekend;
    * `isworkday`: zero distance if day is a weekday;
    * `isworkdayorweekend`: zero distance if both values are either weekday or are weekend;


* LocalTime types:
    * `difference`: simple numerical difference of time in seconds;
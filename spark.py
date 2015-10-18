"""
StackOverflow is a collaboratively edited question-and-answer site generally focused on programming topics. Because of the variety of features tracked, including a variety of feedback metrics, it allows for some open-ended analysis of user behavior on the site.

The data is available on [AWS S3](s3://thedataincubator-course/spark-stack-data/)
There are three subfolders: allUsers, allPosts, and allVotes which contain chunked and gzipped xml with the following format:

``` xml
<row Body="&lt;p&gt;I always validate my web pages, and I recommend you do the same BUT many large company websites DO NOT and cannot validate because the importance of the website looking exactly the same on all systems requires rules to be broken. &lt;/p&gt;&#10;&#10;&lt;p&gt;In general, valid websites help your page look good even on odd configurations (like cell phones) so you should always at least try to make it validate.&lt;/p&gt;&#10;" CommentCount="0" CreationDate="2008-10-12T20:26:29.397" Id="195995" LastActivityDate="2008-10-12T20:26:29.397" OwnerDisplayName="Eric Wendelin" OwnerUserId="25066" ParentId="195973" PostTypeId="2" Score="0" />
```

A full schema can be found [here](https://ia801500.us.archive.org/8/items/stackexchange/readme.txt) which originates from the main archive.org [page](https://archive.org/details/stackexchange).

** Spark Overflow **
We'll use some open-source code (you'll need to clone [this repo](https://github.com/stevenrskelton/SparkOverflow)) to handle the integration of the StackOverflow xml into Scala.

The scala files in the SparkOverflow project will parse the XML for you, allowing you to pull out tags by writing `e.g. x.creationDate`. Look through the scala parsing source code (eg. StackTable.scala, Post.scala) to see how this works. You may also want to look [here](http://stevenskelton.ca/files/2013/12/Real-Time-Data-Mining-With-Spark.scala) and [here](http://stevenskelton.ca/real-time-data-mining-spark/) to get a better grasp of what this code does.

You'll notice that the parsing code (in eg. User.scala) asserts the existence of a xml file - you can remove this requirement to let the code work with chunked data.

The Main.scala file framework can be modified. Here's a place to start - the input and output directory arguments are important for running the automated EMR script.
```
object Main {
  val inputDir = args(0)
  val outputDir = args(1)
  val minSplits = 4

  System.setProperty("spark.executor.memory", "5g")
  System.setProperty("spark.rdd.compress", "true")

  println("Spark starting.")

  val conf = new SparkConf().setAppName("Main")
  conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .set("spark.default.parallelism", "12")
    .set("spark.hadoop.validateOutputSpecs", "false")
  conf.registerKryoClasses(Array(classOf[Post], classOf[User], classOf[Vote]))

  val sc = new SparkContext(conf)
```

It is also important to have this line in your build.sbt - the other dependences in there by default are not necessary for this project.
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.2.0"

The files "project/build.sbt", ".project", and ".classpath" should not be necessary.

** Workflow **
You can use the [scala build tool](www.scala-sbt.org/) to package the scala code into a .jar. The general workflow is
1. edit source code in Main.scala
2. run the command `sbt package` from the root directory of the project
3. use [spark-submit](https://spark.apache.org/docs/latest/submitting-applications.html) locally: this means adding a flag like --master local[2] to the spark-submit command.
4. use the create_spark_cluster script to run spark-submit on EMR

** Tips **
1. It makes sense to do your development on some subset of the entire dataset for the sake of expediency. Data from the much smaller stats.stackexchange.com is available in the same format on [AWS S3](s3://thedataincubator-course/spark-stats-data/)
2. SBT has some nice features, for example [Continuous build and test](http://www.scala-sbt.org/0.12.4/docs/Getting-Started/Running.html#continuous-build-and-test), which can greatly speed up your development.
3. Try eg. cat output_dir/* | sort -n -t , -k 1.2 -o sorted_output to concatenate the various part-xxxxx files.

**Question:** Why do we need to use spark? What are the circmstances where we would favor using this approach over others?
"""

from lib import QuestionList, Question, JsonValidateMixin, TupleListValidateMixin
QuestionList.set_name("spark")

class TupleNumberListValidateMixin(TupleListValidateMixin):
  @classmethod
  def list_length(cls):
    return 100

  @classmethod
  def tuple_validators(cls):
    return (cls.validate_int, cls.validate_number)

@QuestionList.add
class UpvotePercentageByFavorites(TupleNumberListValidateMixin, Question):
  @classmethod
  def list_length(cls):
    return 50
  
  """
  Each post on StackExchange can be upvoted, downvoted, and favorited. One "sanity check" we can do is to look at the ratio of upvotes to downvotes as a function of how many times the post has been favorited.
  Using post favorite counts as the keys for your mapper, calculate the average percentage of upvotes (upvotes / (upvotes + downvotes)) for the first 100 keys (starting from the least favorited posts).
  
  Do the analysis on the stats.stackexchange.com dataset.
  """

  def solution(self):
    """
    A list of 50 tuples of favorite count and upvote percentage.
    """
    return [(20, 0.9952153110047847)] * 50


@QuestionList.add
class UserAnswerPercentageByReputation(TupleNumberListValidateMixin, Question):
  """
  Investigate the correlation between a user's reputation and the kind of posts they make. For the 99 users with the highest reputation, single out posts which are either questions or answers and look at the percentage of their posts that are answers (answers / (answers + questions)).

  You should also return (-1, fraction) to represent the case where you average over all users.

  Again, you only need to run this on the statistics overflow set.
  """

  def solution(self):
    """
    A list of 100 tuples of user IDs and answer percentage (float) using the same format.
    """
    return [(7071,0.9107142857142857)] * 100


@QuestionList.add
class UserReputationByTenure(TupleNumberListValidateMixin, Question):
  """
  If we use the total number of posts made on the site as a metric for tenure, we can look at the differences between "younger" and "older" users. You can imagine there might be many interesting features - for now just return the top 100 post counts and the average reputation for every user who has that count.
  """

  def solution(self):
    """
    A list of 100 tuples of post count and average reputation using the same format.
    """
    return [(118, 3736.5)] * 100


@QuestionList.add
class QuickAnswersByHour(JsonValidateMixin, Question):
  _keys = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

  @classmethod
  def keys(cls):
    return cls._keys

  """
  How long do you have to wait to get your question answered? Look at the set of ACCEPTED answers which are posted less than three hours after question creation. What is the average number of these "quick answers" as a function of the hour of day the question was asked?
  You should normalize by how many total accepted answers are garnered by questions posted in a given hour, just like we're counting how many quick accepted answers are garnered by questions posted in a given hour, eg. (quick accepted answers when question hour is 15 / total accepted answers when question hour is 15).

  Question: What biases are present in our result, that we don't account for? How would we handle this?
  """
  def solution(self):
    """
    Return this json object, values should be floats.
    The average across the entire day is 0.4824340561.
    """
    return {
      "00": 0.,
      "01": 0.,
      "02": 0.,
      "03": 0.,
      "04": 0.,
      "05": 0.,
      "06": 0.,
      "07": 0.,
      "08": 0.,
      "09": 0.,
      "10": 0.,
      "11": 0.,
      "12": 0.,
      "13": 0.,
      "14": 0.,
      "15": 0.,
      "16": 0.,
      "17": 0.,
      "18": 0.,
      "19": 0.,
      "20": 0.,
      "21": 0.,
      "22": 0.,
      "23": 0.
    }


@QuestionList.add
class QuickAnswersByHourFull(JsonValidateMixin, Question):
  _keys = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]

  @classmethod
  def keys(cls):
    return cls._keys

  def solution(self):
    """
    Same as above, but on the full StackOverflow dataset.
    The average over the entire day is 0.7231554846.
    """
    return {
      "00": 0.,
      "01": 0.,
      "02": 0.,
      "03": 0.,
      "04": 0.,
      "05": 0.,
      "06": 0.,
      "07": 0.,
      "08": 0.,
      "09": 0.,
      "10": 0.,
      "11": 0.,
      "12": 0.,
      "13": 0.,
      "14": 0.,
      "15": 0.,
      "16": 0.,
      "17": 0.,
      "18": 0.,
      "19": 0.,
      "20": 0.,
      "21": 0.,
      "22": 0.,
      "23": 0.
    }


@QuestionList.add
class IdentifyVeteransFromFirstPostStats(JsonValidateMixin, Question):
  _keys = ["vet_views", "vet_score", "vet_favorites", "vet_answers", "brief_views", "brief_score", "brief_favorites", "brief_answers"]

  @classmethod
  def keys(cls):
    return cls._keys

  """
  It can be interesting to think about what factors influence a user to remain active on the site over a long period of time.
  In order not to bias the results towards older users, we'll define a time window between 100 and 150 days after account creation. If the user has made a post in this time, we'll consider them active and well on their way to being veterans of the site; if not, they are inactive and were likely brief users.

  Question: What other parameterizations of "activity" could we use, and how would they differ in terms of splitting our user base?
  Question: What other biases are still not dealt with, after using the above approach?

  Let's see if there are differences between the first ever question posts of "veterans" vs. "brief users". For each group separately, average the score, views, number of answers, and number of favorites of users' first question.
  Question: What story could you tell from these numbers? How do the numbers support it?

  *Checkpoint*
  The reference solution finds 2,027 veteran users and 24,864 brief users in the stats dataset.
  For the full stackoverflow dataset, it finds 288,285 veterans and 1,848,628 brief users.
  """
  def solution(self):
    """
    Return this json object, values should be floats.
    """
    return {
      "vet_views": 0.,
      "vet_score": 0.,
      "vet_favorites": 0.,
      "vet_answers": 0.,
      "brief_views": 0.,
      "brief_score": 0.,
      "brief_favorites": 0.,
      "brief_answers": 0.
    }


@QuestionList.add
class IdentifyVeteransFromFirstPostStatsFull(JsonValidateMixin, Question):
  _keys = ["vet_views", "vet_score", "vet_favorites", "vet_answers", "brief_views", "brief_score", "brief_favorites", "brief_answers"]

  @classmethod
  def keys(cls):
    return cls._keys

  def solution(self):
    """
    Same thing on the full StackOverflow dataset.
    """
    return {
      "vet_views": 0.,
      "vet_score": 0.,
      "vet_favorites": 0.,
      "vet_answers": 0.,
      "brief_views": 0.,
      "brief_score": 0.,
      "brief_favorites": 0.,
      "brief_answers": 0.
    }

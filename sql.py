# -*- coding: utf-8-*-

from numbers import Number

from lib import QuestionList, Question, TupleListValidateMixin, catch_validate_exception
QuestionList.set_name("sql")

"""
The city of New York does restaurant inspections and assigns a grade.  Inspections data the last 4 years are available [here](https://s3.amazonaws.com/thedataincubator/coursedata/nyc_inspection_data.zip).

The file `RI_Webextract_BigApps_Latest.xls` contains a description of each of the datafiles.  Take a look and then load the csv formatted `*.txt` files into Postgresql into five tables:
1. `actions`
2. `cuisines`
3. `violations`
4. `grades` (from `WebExtract.txt`)
5. `boroughs` (from `RI_Webextract_BigApps_Latest.xls`)

**Hints:**
1. It is recommended to use sqlite3 for this project. Postgresql can work but will be more difficult to set up properly on Digital Ocean. If you do use sqlite, in order to do mathematical calculations like square root, you will need to compile and install the extension described on the [wiki](https://sites.google.com/a/thedataincubator.com/the-data-incubator-wiki/course-information-and-logistics/getting-started/setup).

2. Postgresql has a [`\copy` command](http://www.postgresql.org/docs/9.2/static/app-psql.html#APP-PSQL-META-COMMANDS-COPY) that can both save and load files in various formats.  It is a convenience wrapper for the [`copy` command](http://www.postgresql.org/docs/9.2/static/sql-copy.html) but behaves better (e.g. relative paths).

3. The files may contain malformatted text.  Unfortunately, this is all too common.  As a stop gap, remember that `iconv` is a unix utility that can convert files between different text encodings.

4. For more sophisticated needs, a good strategy is to write simple python scripts that will reparse files.  For example, commas (',') within a single field will trick many csv parsers into breaking up the field.  Write a python script that converts these 'inadvertent' delimiters into semicolons (';').
"""

class GroupbyValidateMixin(TupleListValidateMixin):
  @classmethod
  def list_length(cls):
    return cls._list_length

  @classmethod
  def tuple_validators(cls):
    return (
      cls.validate_string,
      cls.validate_float,
      cls.validate_float,
      cls.validate_int
    )


@QuestionList.add
class ScoreByZipcode(GroupbyValidateMixin, Question):
  """
  Return a list of tuples of the form
  `(zipcode, mean grade, standard error, number of inspections)`
  for each of the 183 zipcodes in the city with over 100 inspections.  You can read more about standard error on [wikipedia](http://en.wikipedia.org/wiki/Standard_error).  Sort the list in ascendig order by score.
  """
  _list_length = 183

  def solution(self):
    return [("11201", 21.9060928719313812, 0.179441607823702, 6762)] * 183


@QuestionList.add
class ScoreByMap(Question):
  """
  The above are not terribly enlightening.  Use [CartoDB](http://cartodb.com/) to produce a map of average scores by zip code.  You can sign up for a free trial.

  You will have to use their wizard to plot the data by [zipcode](http://docs.cartodb.com/cartodb-editor.html#geocoding-data).  Then use the "share" button to return a link to a short URL beginning with "http://cdb.io/".

  **For fun:** How do JFK, Brighton Beach, Liberty Island (home of the Statue of Liberty), Financial District, Chinatown, and Coney Island fare?
  """
  def solution(self):
    return "http://cdb.io/....."

  @catch_validate_exception
  def validate(self):
    solution = self.solution()
    if not isinstance(solution, basestring):
      return "Expected a string but got %.50s" % str(solution)

    if not solution.startswith("http://cdb.io/"):
      return "Expected a cartodb link that starts with http://cdb.io"

    return None


@QuestionList.add
class ScoreByBorough(GroupbyValidateMixin, Question):
  """
  Return a list of tuples of the form
  `(borough, mean grade, stderr, number of inspections)`
  for each of the city's five boroughs.  **Hint**: you will have to perform a join with the `boroughs` table.  Sort the list in ascendig order by score.
  `
  """
  _list_length = 5

  def solution(self):
    return [("MANHATTAN", 22.2375933589636849, 0.0332739265922062, 204185)] * 5


@QuestionList.add
class ScoreByCuisine(GroupbyValidateMixin, Question):
  """
  Return a list of the 75 tuples of the form
  `(cuisine, mean grade, stderr, number of inspections)`
  for each of the 75 cuisine types with at least 100 inspections.  **Hint**: you will have to perform a join with the `boroughs` table.  Sort the list in ascendig order by score.  Are the least sanitary and most sanitary cuisine types surprising?
  """
  _list_length = 75

  def solution(self):
    return [("French", 21.9985734664764622, 0.177094690841052, 7010)] * 75


@QuestionList.add
class ViolationByCuisine(TupleListValidateMixin, Question):
  """
  Which cuisines tend to have a disproportionate number of what which violations?  Answering this question isn't easy becuase you have to think carefully about normalizations.

  1. More popular cuisine categories will tend to have more violations just becuase they represent more restaurants.

  2. Similarly, some violations are more common.  For example, knowing that "Equipment not easily movable or sealed to floor" is a common violation for Chinese restuarants is not particularly helpful when it is a common violation for all restaurants.

  The right quantity is to look at is the conditional probability of a specific type of violation given a specific cuisine type and divide it by the unconditional probability of the violation for the entire population.  Taking this ratio gives the right answer.  Return the 20 highest ratios.

  **Hint**:
  1. You might want to check out this [Stackoverflow post](http://stackoverflow.com/questions/972877/calculate-frequency-using-sql).

  2. The definition of a violation changes with time.  For example, 10A can mean two different things "Toilet facility not maintained ..." or "Vermin or other live animal present ..." when things were prior to 2003.

  3. The ratios don't mean much when the number of violations of a given type and for a specific category are not large (why not?).  Be sure to filter these out.  We chose 100 as our cutoff.
  """
  def solution(self):
    return [(("Café/Coffee/Tea", "Toilet facility not maintained and provided with toilet paper; waste receptacle and self-closing door."), 1.8836420929815939, 315)] * 20

  @classmethod
  def list_length(cls):
    return 20

  @classmethod
  def tuple_validators(cls):
    return (
      cls.validate_tuple,
      cls.validate_float,
      cls.validate_int,
    )



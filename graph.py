"""
[New York Social Diary](http://www.newyorksocialdiary.com/) provides a fascinating lens onto New York's socially well-to-do.  The data forms a natural social graph for New York's social elite.  Take a look at this page of a recent run-of-the-mill holiday party:

`http://www.newyorksocialdiary.com/party-pictures/2014/holiday-dinners-and-doers`

Besides the brand-name celebrities, you will notice the photos have carefully annotated captions labeling those that appear in the photos.  We can think of this as implicitly implying a social graph: there is a connection between two individuals if they appear in a picture together.

The first step is to fetch the data.  This comes in two phases.

The first step is to crawl the data.  We want photos from parties before December 1st, 2014.  Go to
`http://www.newyorksocialdiary.com/party-pictures`
to see a list of (party) pages.  For each party's page, grab all the captions.

*Checkpoint*
The reference solution found 1262 total pages in the archive (without date filtering).

*Hints*:

  1. Click on the on the index page and see how they change the url.  Use this to determine a strategy to get all the data.

  2. Notice that each party has a date on the index page.  Use python's `datetime.strptime` function to parse it.

  3. Some captions are not useful: they contain long narrative texts that explain the event.  Usually in two stage processes like this, it is better to keep more data in the first stage and then filter it out in the second stage.  This makes your work more reproducible.  It's usually faster to download more data than you need now than to have to redownload more data later.

Now that you have a list of all captions, you should probably save the data on disk so that you can quickly retrieve it.  Now comes the parsing part.

  1. Some captions are not useful: they contain long narrative texts that explain the event.  Try to find some heuristic rules to separate captions that are a list of names from those that are not.  A few heuristics include:
      - look for sentences (which have verbs) and as opposed to lists of nouns.  For example, [nltk does part of speech tagging](http://www.nltk.org/book/ch05.html) but it is a little slow.  There may also be heuristics that accomplish the same thing.
      - Look for commonly repeated threads (e.g. you might end up picking up the photo credtis).
      - Long captions are often not lists of people.  The cutoff is subjective so to be definitive, *let's set that cutoff at 250 characters*.

  2. You will want to separate the captions based on various forms of punctuation.  Try using `re.split`, which is more sophisticated than `string.split`.

  3. You might find a person named "ra Lebenthal".  There is no one by this name.  Can anyone spot what's happening here?

  4. This site is pretty formal and likes to say things like "Mayor Michael Bloomberg" after his election but "Michael Bloomberg" before his election.  Can you find other ('optional') titles that are being used?  They should probably be filtered out b/c they ultimately refer to the same person: "Michael Bloomberg."

For the analysis, we think of the problem in terms of a [network](http://en.wikipedia.org/wiki/Computer_network) or a [graph](http://en.wikipedia.org/wiki/Graph_%28mathematics%29).  Any time a pair of people appear in a photo together, that is considered a link.  What we have described is more appropriately called an (undirected) [multigraph](http://en.wikipedia.org/wiki/Multigraph) with no self-loops but this has an obvious analog in terms of an undirected [weighted graph](http://en.wikipedia.org/wiki/Graph_%28mathematics%29#Weighted_graph).  In this problem, we will analyze the social graph of the new york social elite.

For this problem, we recommend using python's `networkx` library.
"""

from lib import QuestionList, Question, StringNumberListValidateMixin, TupleListValidateMixin
QuestionList.set_name("graph")

@QuestionList.add
class Degree(StringNumberListValidateMixin, Question):
  """
  The simplest question you might want to ask is 'who is the most popular'?  The easiest way to answer this question is to look at how many connections everyone has.  Return the top 100 people and their degree.  Remember that if an edge of the graph has weight 2, it counts for 2 in the degree.
  """
  def solution(self):
    """
    A list of 100 tuples of (name, degree) in descending order of degree
    Overall solution stats:
    Number of nodes: 102261
    Number of edges: 191926
    Average degree:   3.7536

    Top 100 .describe()
    count    100.000000
    mean     106.340000
    std       51.509579
    min       69.000000
    25%       77.000000
    50%       85.500000
    75%      116.500000
    max      372.000000
    """
    return [('Jean Shafiroff', 605),('Mark Gilbertson', 486),('Gillian Miniter', 424),('Alexandra Lebenthal', 394),('Geoffrey Bradfield', 369),('Debbie Bancroft', 289),('Andrew Saffir', 286),('Sharon Bush', 283),('Somers Farkas', 279),('Michael Bloomberg', 269),('Alina Cho', 257),('Kamie Lightburn', 257),('Eleanora Kennedy', 247),('Muffie Potter Aston', 247),('Bonnie Comley', 233),('Lydia Fenet', 232),('Allison Aston', 229),('Mario Buatta', 226),('Lucia Hwong Gordon', 220),('Liliana Cavendish', 220),('Deborah Norville', 218),('Patrick McMullan', 206),('Bettina Zilkha', 204),('Jamee Gregory', 202),('Karen LeFrak', 200),('Stewart Lane', 194),('Nicole Miller', 186),('Daniel Benedict', 180),('Evelyn Lauder', 176),('Dennis Basso', 176),('Leonard Lauder', 176),('Karen Klopp', 175),('Elizabeth Stribling', 174),('Diana Taylor', 174),('Roric Tobin', 173),('Christopher Hyland', 171),('Paula Zahn', 170),('Martha Stewart', 166),('Amy Fine Collins', 166),('Amy McFarland', 166),('Fernanda Kellogg', 166),('Jennifer Creel', 166),('Couri Hay', 166),('Kipton Cronkite', 164),('Barbara Tober', 162),('Margo Langenberg', 157),('Alexandra Lind Rose', 156),('Tory Burch', 153),('Audrey Gruss', 153),('Nathalie Kaplan', 152),('Liz Peek', 151),('Donna Karan', 149),('Gregory Long', 149),('Amy Hoadley', 146),('Adelina Wong Ettelson', 146),('Anne Hearst McInerney', 145),('Russell Simmons', 145),('Hilary Geary Ross', 144),('Rosanna Scotto', 144),('Grace Meigher', 141),('Dayssi Olarte de Kanavos', 140),('Janna Bullock', 140),('Fe Fendi', 139),('Wilbur Ross', 139),('Margaret Russell', 139),('Martha Glass', 139),('Deborah Roberts', 138),('Margo Catsimatidis', 138),('Board Member', 138),('Alexia Hamm Ryan', 138),('Anka Palitz', 137),('Diana DiMenna', 136),('Felicia Taylor', 136),('David Koch', 135),('Barbara Regna', 135),('Susan Shin', 134),('Clare McKeon', 133),('Frederick Anderson', 133),('Susan Magazine', 133),('Hunt Slonem', 132),('Gerald Loughlin', 132),('Annette Rickel', 131),('Lisa Anastos', 131),('Bette Midler', 130),('Coralie Charriol Paul', 129),('Jay McInerney', 128),('Dawne Marie Grannum', 126),('Fern Mallis', 126),('Michele Gerber Klein', 125),('Alec Baldwin', 125),('Karen Pearl', 123),('Wendy Carduner', 122),('Pamela Fiori', 121),('Cynthia Lufkin', 121),('Nina Griscom', 118),('Beth Rudin DeWoody', 118),('Steven Stolman', 117),('Chris Meigher', 117),('Allison Mignone', 117),('Patricia Shiah', 116)]


@QuestionList.add
class PageRank(StringNumberListValidateMixin, Question):
  """
  A similar way to determine popularity is to look at their [pagerank](http://en.wikipedia.org/wiki/PageRank).  Pagerank is used for web ranking and was originally [patented](http://patft.uspto.gov/netacgi/nph-Parser?patentnumber=6285999) by Google and is essentially the [stationary distribution](http://en.wikipedia.org/wiki/Markov_chain#Stationary_distribution_relation_to_eigenvectors_and_simplices) of a [markov chain](http://en.wikipedia.org/wiki/Markov_chain) implied by the social graph.

  Use 0.85 as the damping parameter so that there is a 15% chance of jumping to another vertex at random.
  """
  def solution(self):
    """
    A list of 100 tuples of the form (name, pagerank) in descending order of pagerank

    Topp 100 .describe()
    count    100.000000
    mean       0.000185
    std        0.000076
    min        0.000124
    25%        0.000138
    50%        0.000162
    75%        0.000200
    max        0.000623
    """
    return [('Jean Shafiroff', 0.0006721583166126367),('Mark Gilbertson', 0.0005300640771106098),('Gillian Miniter', 0.0004328564749382346),('Geoffrey Bradfield', 0.00039445609881565933),('Alexandra Lebenthal', 0.0003779383781528194),('Andrew Saffir', 0.00034672762831650995),('Sharon Bush', 0.00031566275316038066),('Michael Bloomberg', 0.0003115562934130505),('Mario Buatta', 0.00030594514265875136),('Debbie Bancroft', 0.0002993967044320434),('Somers Farkas', 0.00029260287213965914),('Kamie Lightburn', 0.00028095460279265225),('Alina Cho', 0.0002785748306272263),('Patrick McMullan', 0.00027350210675991917),('Lucia Hwong Gordon', 0.0002687769492090694),('Christopher Hyland', 0.0002518918081726786),('Lydia Fenet', 0.00024945308741560244),('Liliana Cavendish', 0.00024814775976812227),('Bonnie Comley', 0.00024701816358534514),('Eleanora Kennedy', 0.00024532916248425513),('Barbara Tober', 0.00023024458256419212),('Muffie Potter Aston', 0.00022873858610252158),('Deborah Norville', 0.0002238493536119899),('Bettina Zilkha', 0.00022193831919351318),('Allison Aston', 0.0002182621480887853),('Martha Stewart', 0.0002166496415565429),('Karen LeFrak', 0.00021652463392615831),('Elizabeth Stribling', 0.0002144431356579515),('Jamee Gregory', 0.00021103549552795966),('Nicole Miller', 0.00021099509795761978),('Fernanda Kellogg', 0.0002091440218451554),('Donna Karan', 0.0002091175063763339),('Steven Stolman', 0.00020724621061613),('Amy Fine Collins', 0.00020716270929498401),('Kipton Cronkite', 0.00020694753465622288),('Karen Klopp', 0.00020233211624856468),('Leonard Lauder', 0.0002013640704623924),('Diana Taylor', 0.0001959912329944465),('Couri Hay', 0.00019597750786643986),('Alec Baldwin', 0.0001952837197763471),('Russell Simmons', 0.00019509973837435388),('Daniel Benedict', 0.00019348823904852757),('Stewart Lane', 0.00019322843286425506),('Dennis Basso', 0.0001929377378808544),('Margo Langenberg', 0.00019227641912869382),('Evelyn Lauder', 0.00018776268260901782),('Dawne Marie Grannum', 0.00018348446309129562),('Anka Palitz', 0.0001816349294641944),('Roric Tobin', 0.0001792033476270693),('Jennifer Creel', 0.000176650852661298),('Paula Zahn', 0.00017405784750202177),('Michele Gerber Klein', 0.00017391044576958088),('Audrey Gruss', 0.0001732707939127245),('Amy McFarland', 0.00017253299814631428),('Fern Mallis', 0.00017214996711406935),('Rosanna Scotto', 0.0001718239179294061),('Susan Shin', 0.00017122598412544065),('Gregory Long', 0.00017075845174831323),('Barbara Regna', 0.00017020702026546115),('Bette Midler', 0.0001692069101900205),('Lisa Anastos', 0.00016839624128551828),('Annette Rickel', 0.0001652984634064357),('Tory Burch', 0.00016527253466787597),('Nathalie Kaplan', 0.0001638399205760537),('Janna Bullock', 0.00016197813503127694),('Margaret Russell', 0.00016032155700348365),('Amy Hoadley', 0.00016003634133206413),('Deborah Roberts', 0.00016002374977805024),('Liz Peek', 0.0001593432914719983),('Georgina Schaeffer', 0.00015807720955121482),('Pamela Fiori', 0.00015748887772618097),('Agnes Gund', 0.00015739079199585714),('Grace Meigher', 0.00015735855778809975),('Dayssi Olarte de Kanavos', 0.00015722725538680225),('Kristian Laliberte', 0.00015663154153108867),('Felicia Taylor', 0.00015600583831208584),('Adelina Wong Ettelson', 0.00015447989420804914),('Alexandra Lind Rose', 0.0001526729194305092),('Beth Rudin DeWoody', 0.0001521752757062377),('Hunt Slonem', 0.00015034573024649012),('Karen Pearl', 0.00015014270968139078),('Cassandra Seidenfeld', 0.00014978411504264205),('Fe Fendi', 0.0001479056176433673),('Richard Johnson', 0.00014737570331573567),('Tina Brown', 0.00014736230342231722),('Mary Van Pelt', 0.00014615150332793816),('Denise Rich', 0.00014477512756538359),('Margo Catsimatidis', 0.0001446019750237962),('Amy Phelan', 0.00014361917109864106),('Martha Glass', 0.00014329423430215027),('Anne Hearst McInerney', 0.00014146806305781302),('Alexia Hamm Ryan', 0.00014035117069104687),('Hoda Kotb', 0.00013984454946513834),('Jamie Niven', 0.00013968205111474316),('Hilary Geary Ross', 0.00013922637330495693),('Bunny Williams', 0.00013910688252452672),('Clare McKeon', 0.00013765028672167918),('David Koch', 0.000137221960819086),('Tinsley Mortimer', 0.00013695769601492294),('Wendy Carduner', 0.00013670908833013031)]


@QuestionList.add
class BestFriends(TupleListValidateMixin, Question):
  """
  Another interesting question is who tend to co-occur with each other.  Give us the 100 edges with the highest weights

  Google these people and see what their connection is.  Can we use this to detect instances of infidelity?
  """
  def solution(self):
    """
    A list of 100 tuples of the form ((person1, person2), count) in descending order of count

    Topp 100 .describe()
    count    100.000000
    mean      25.070000
    std       15.647154
    min       13.000000
    25%       15.000000
    50%       19.000000
    75%       28.500000
    max      107.000000
    """
    return [(('Bonnie Comley', 'Stewart Lane'), 77),(('Geoffrey Bradfield', 'Roric Tobin'), 68),(('Andrew Saffir', 'Daniel Benedict'), 64),(('Jay Diamond', 'Alexandra Lebenthal'), 51),(('Elizabeth Stribling', 'Guy Robinson'), 37),(('Deborah Norville', 'Karl Wellner'), 35),(('Hilary Geary Ross', 'Wilbur Ross'), 34),(('Fernanda Kellogg', 'Kirk Henckels'), 33),(('Frederick Anderson', 'Douglas Hannant'), 31),(('Janna Bullock', 'Couri Hay'), 31),(('Anne Hearst McInerney', 'Jay McInerney'), 28),(('Leonel Piraino', 'Nina Griscom'), 28),(('Michael Cominotto', 'Dennis Basso'), 27),(('Arlene Dahl', 'Marc Rosen'), 26),(('Mark Badgley', 'James Mischka'), 26),(('Olivia Palermo', 'Johannes Huebl'), 25),(('Sylvester Miniter', 'Gillian Miniter'), 24),(('Hunt Slonem', 'Liliana Cavendish'), 23),(('Al Roker', 'Deborah Roberts'), 22),(('Margo Catsimatidis', 'John Catsimatidis'), 22),(('Sherrell Aston', 'Muffie Potter Aston'), 22),(('David Lauren', 'Lauren Bush'), 21),(('Michael Kovner', 'Jean Doyen de Montaillou'), 21),(('Diana Taylor', 'Michael Bloomberg'), 21),(('Rick Friedberg', 'Francine LeFrak'), 21),(('Stephanie Krieger', 'Brian Stewart'), 20),(('Sharon Bush', 'Jean Shafiroff'), 20),(('Nicole Miller', 'Kim Taipale'), 19),(('Jean Shafiroff', 'Patricia Shiah'), 18),(('Jonathan Tisch', 'Lizzie Tisch'), 18),(('Tony Ingrao', 'Randy Kemper'), 17),(('Alina Cho', 'John Demsey'), 17),(('Alexandra Lebenthal', 'Gillian Miniter'), 17),(('Will Cotton', 'Rose Dergan'), 17),(('Susan Magazine', 'Nicholas Scoppetta'), 17),(('Anna Safir', 'Eleanora Kennedy'), 17),(('Somers Farkas', 'Jonathan Farkas'), 17),(('Somers Farkas', 'Muffie Potter Aston'), 17),(('Bunny Williams', 'John Rosselli'), 16),(('Robert Bradford', 'Barbara Taylor Bradford'), 16),(('Bonnie Comley', 'Leah Lane'), 15),(('Jean Shafiroff', 'Lucia Hwong Gordon'), 15),(('Jean Shafiroff', 'Martin Shafiroff'), 15),(('Gillian Hearst Simonds', 'Christian Simonds'), 15),(('Daniel Benedict', 'Johannes Huebl'), 15),(('Delfina Blaquier', 'Nacho Figueras'), 14),(('Nick Korniloff', 'Pamela Cohen'), 14),(('Grace Meigher', 'Chris Meigher'), 14),(('Campion Platt', 'Tatiana Platt'), 14),(('Edward Callaghan', 'John Wegorzewski'), 14),(('Charlotte Ronson', 'Ali Wise'), 14),(('Stewart Lane', 'Leah Lane'), 14),(('Debbie Bancroft', 'Tiffany Dubin'), 14),(('Geoffrey Thomas', 'Sharon Sondes'), 14),(('Ann Rapp', 'Roy Kean'), 14),(('Larry Wohl', 'Leesa Rowland'), 13),(('Darci Kistler', 'Peter Martins'), 13),(('Edwina Sandys', 'Richard Kaplan'), 13),(('Ken Starr', 'Diane Passage'), 13),(('Richard Farley', 'Chele Chiavacci'), 13),(('Elyse Newhouse', 'Gillian Miniter'), 13),(('Caroline Murphy', 'Heather Matarazzo'), 13),(('Donald Tober', 'Barbara Tober'), 13),(('Gary Lawrance', 'Zita Davisson'), 13),(('Desiree Gruber', 'Kyle MacLachlan'), 13),(('Yasmin Aga Khan', 'Blaise Labriola'), 12),(('Clare McKeon', 'Lydia Fenet'), 12),(('Margo Langenberg', 'Jean Shafiroff'), 12),(('Simon Doonan', 'Jonathan Adler'), 12),(('Judy Licht', 'Jerry Della Femina'), 12),(('Sharon Bush', 'Ashley Bush'), 12),(('George Hambrecht', 'Andrea Fahnestock'), 12),(('Lorenzo Martone', 'Marc Jacobs'), 12),(('Russell Simmons', 'Porschla Coleman'), 12),(('Sharyn Mann', 'Todd Slotkin'), 12),(('Katharina Otto', 'Nathan Bernstein'), 12),(('Anna Wintour', 'Bee Shaffer'), 12),(('Harry Slatkin', 'Laura Slatkin'), 12),(('Julianna Margulies', 'Keith Lieberthal'), 12),(('Maureen Chilton', 'Gregory Long'), 12),(('Candace Bushnell', 'Charles Askegard'), 12),(('Howard Sobel', 'Gayle Sobel'), 12),(('Tom McCarter', 'Frances Scaife'), 11),(('Brad Learmonth', 'Jon Gilman'), 11),(('Lisa McCarthy', 'Libby Fitzgerald'), 11),(('Susan Krysiewicz', 'Thomas Bell'), 11),(('John Connolly', 'Ingrid Connolly'), 11),(('Anne Ford', 'Charlotte Ford'), 11),(('Fe Fendi', 'Patricia Shiah'), 11),(('Eric Villency', 'Caroline Fare'), 11),(('Sandra Brant', 'Ingrid Sischy'), 11),(('Mish Tworkowski', 'Joseph Singer'), 11),(('Diana Taylor', 'Ana Oliveira'), 11),(('Celerie Kemble', 'Boykin Curry'), 11),(('Geoffrey Bradfield', 'Sue Chalom'), 11),(('Geoffrey Bradfield', 'Helena Lehane'), 11),(('Jay McInerney', 'Anne Hearst'), 11),(('Urban Karlsson', 'Juan Montoya'), 11),(('Samuel Waxman', 'Marion Waxman'), 11),(('Jay Johnson', 'Tom Cashin'), 11)]

  def list_length(cls):
    return 100

  @classmethod
  def tuple_validators(cls):
    return (cls.validate_tuple, cls.validate_int)

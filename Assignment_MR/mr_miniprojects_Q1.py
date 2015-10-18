from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import heapq
WORD_RE = re.compile(r"\w+")

class MR_Q1(MRJob):
        
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(mapper_init = self.heap_init,
                   mapper = self.mapper_add_to_head,
                   mapper_final = self.mapper_pop_top_100,
                   reducer_init = self.reducer_heap_init,
                   reducer = self.reducer_heap_count_words,
                   reducer_final = self.reducer_pop_top_100)]
        

    def mapper_get_words(self, _, line):
        # yield each word in the line
        for word in WORD_RE.findall(line):
            yield (word.lower(), 1)

    def combiner_count_words(self, word, counts):
        # optimization: sum the words we've seen so far
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        # optimization: sum the words we've seen so far
        yield (word, sum(counts))

    def heap_init(self):
        self.h = []
        self.h_reducer = []
        
    def mapper_add_to_head(self,word,counts):
        heapq.heappush(self.h,(counts,word))
        #yield (word,counts)
        
    def mapper_pop_top_100(self):
        #print 'POP TOP 100 MAPPER'
        largest = heapq.nlargest(100,self.h)
        
        for count,word in largest:
            yield ('heap',(count,word))  
            
    def reducer_heap_init(self):
        self.h_unused = []
        
    def reducer_heap_count_words(self, key,word_counts):
        # optimization: sum the words we've seen so far
        for wc in word_counts:
            heapq.heappush(self.h_unused, (wc[0],wc[1]))
    
    def reducer_pop_top_100(self):
        largest = heapq.nlargest(100,self.h_unused)
        words = [(word,  int(count)) for count,word in largest]
        yield (None, words)      
        #for count,word in largest:
        #    yield (word,count)

            
        
if __name__ == '__main__':
    MR_Q1.run()
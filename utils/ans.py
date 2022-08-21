"""ans"""
import numpy as np


def first_1_index(val):
    """Return the Index of the First Non-Zero Bit."""
    counter = 0
    while val > 1:
        counter += 1
        val = val >> 1
    return counter


def output_nb_bits(state, nb_bits):
    """Output NbBits to a BitStream"""
    mask = (1 << nb_bits) - 1
    little = state & mask
    if nb_bits > 0:
        string = "{:b}".format(little)
    else:
        return ""
    while len(string) < nb_bits:
        string = "0" + string
    return string


def bits_to_state(bit_stream, nb_bits):
    """ Convert Bits from Bitstream to the new State. """
    if nb_bits == 0 or len(bit_stream) == 0:
        return 0, bit_stream
    if nb_bits == len(bit_stream):
        rest = int(bit_stream, 2)
        return rest, ""
    bits = bit_stream[-nb_bits:]
    rest = int(bits, 2)
    remaining = bit_stream[:-nb_bits]
    return rest, remaining


def decode_symbol(state, bit_stream, state_t):
    """ Return a Symbol + New State + Bitstream from the bitStream and State. """
    symbol = state_t[state]['symbol']
    nb_bits = state_t[state]['nbBits']
    rest, bit_stream = bits_to_state(bit_stream, nb_bits)
    state = state_t[state]['newX'] + rest
    return symbol, state, bit_stream


class TabledANS:
    def __init__(self, symbol_occurrences, table_log=5):
        self.table_log = table_log
        self.table_size = 1 << table_log
        if self.table_size < len(symbol_occurrences):
            raise RuntimeError("Table size {} less than number of symbols {}"
                               .format(self.table_size, len(symbol_occurrences)))
        freq_sum = np.sum(list(symbol_occurrences.values()))
        #print(len(symbol_occurrences.values()), freq_sum)
        #print(symbol_occurrences.values())
        if freq_sum != self.table_size:
            # Normalize frequencies table
            freq_norm = \
                np.array([max(1, np.round(self.table_size * symbol_occurrences[sym] / freq_sum))
                          for sym in symbol_occurrences.keys()])
            freq_sum_norm = np.sum(freq_norm)
            reminder = self.table_size - freq_sum_norm
            #print(reminder)
            while reminder < 0:
                #shrink the frequencies to fit the table
                max_ix = np.argmax(freq_norm)
                freq_norm[max_ix] -= 1
                reminder += 1
                #delta = reminder // freq_norm[freq_norm > 1].size
                #for i in range(len(freq_norm)):
                #    if reminder == 0:
                #        break
                #    if freq_norm[i] > 1:
                #        freq_norm[i] += delta if reminder <= delta else reminder
                #        reminder -= delta if reminder <= delta else reminder
            if reminder > 0:
                #grow the frequencies to fit the table
                max_ix = np.argmax(freq_norm)
                freq_norm[max_ix] += reminder

            #elif reminder > 1:
                #for i in range(len(freq_norm)):
                    #if reminder == 0:
                        #break
                    #freq_norm[i] += 1
                    #reminder -= 1
            #print(freq_norm.sum())
            assert freq_norm.sum() == self.table_size
            symbol_occurrences = dict([(k, int(freq_norm[i]))
                                       for i, k in enumerate(symbol_occurrences.keys())])
        ####
        # Define the Initial Positions of States in StateList.
        ####
        symbol_list = [symbol for symbol, occcurences in symbol_occurrences.items()]
        cumulative = [0 for _ in range(len(symbol_list)+2)]
        for u in range(1, len(symbol_occurrences.items())+ 1):
            cumulative[u] = cumulative[u - 1] + list(symbol_occurrences.items())[u-1][1]
        cumulative[-1] = self.table_size + 1
        #####
        # Spread Symbols to Create the States Table
        #####
        high_thresh = self.table_size - 1
        state_table = [0 for _ in range(self.table_size)]
        table_mask = self.table_size - 1
        step = ((self.table_size >> 1) + (self.table_size >> 3) + 3)
        pos = 0
        for symbol, occurrences in symbol_occurrences.items():
            for i in range(occurrences):
                state_table[pos] = symbol
                pos = (pos + step) & table_mask
                #while pos > high_thresh: print("Huuuh") position = (pos + step) & table_mask
        assert pos == 0
        #####
        # Build Coding Table from State Table
        #####
        #outputBits = [0 for _ in range(self.tableSize)]
        self.coding_table = [0 for _ in range(self.table_size)]
        cumulative_cp = cumulative.copy()
        for i in range(self.table_size):
            s = state_table[i]
            index = symbol_list.index(s)
            self.coding_table[cumulative_cp[index]] = self.table_size + i
            cumulative_cp[index] += 1
            #outputBits[i] = self.tableLog - first1Index(self.tableSize + i)
        #print(freq_norm)
        #print("skip table", step)
        #print(state_table - min(state_table))
        #####
        # Create the Symbol Transformation Table
        #####
        total = 0
        self.symbol_tt = {}
        for symbol, occurrences in symbol_occurrences.items():
            self.symbol_tt[symbol] = {}
            if occurrences == 1:
                self.symbol_tt[symbol]['deltaNbBits'] = (self.table_log << 16) - (1 << self.table_log)
                self.symbol_tt[symbol]['deltaFindState'] = total - 1
            elif occurrences > 0:
                max_bits_out = self.table_log - first_1_index(occurrences - 1)
                min_state_plus = occurrences << max_bits_out
                self.symbol_tt[symbol]['deltaNbBits'] = (max_bits_out << 16) - min_state_plus
                self.symbol_tt[symbol]['deltaFindState'] = total - occurrences
            total += occurrences
        #print("deltaNbBits of symbol ", symbol, " is ", self.symbolTT[symbol]['deltaNbBits'])
        # print(self.symbolTT)
        #####
        # Generate a Decoding Table
        #####
        self.decode_table = [{} for _ in range(self.table_size)]
        nextt = list(symbol_occurrences.items())
        for i in range(self.table_size):
            t = {}
            t['symbol'] = state_table[i]
            index = symbol_list.index(t['symbol'])
            x = nextt[index][1]
            nextt[index] = (nextt[index][0], nextt[index][1] + 1)
            t['nbBits'] = self.table_log - first_1_index(x)
            t['newX'] = (x << t['nbBits']) - self.table_size
            self.decode_table[i] = t
            #print(t['symbol'] - min(state_table), t['nbBits'], t['newX'])
        #print("decodeTable size is ", len(self.decodeTable))


    @staticmethod
    def from_data(data, table_log=None):
        """from data"""
        #c, v =
        # np.histogram(data, bins=int(max(data)-min(data))+1, range= [min(data)-.5, max(data)+.5])
        #v = (v+0.5)[:-1]
        v, c = np.unique(data, return_counts=True)
        symbol_occurrences = dict([(v_, c_) for v_, c_ in zip(v, c)])
        if table_log is None:
            table_log = max(5, 3 + int(np.ceil(np.log2(len(c))))) # sefi added
        return TabledANS(symbol_occurrences, table_log)

    def encode_efficient(self, symbol, state, symbol_tt):
        """encode efficient"""
        symbol_tt = symbol_tt[symbol]
        nb_bits_out = (state + symbol_tt['deltaNbBits']) >> 16
        eff = output_nb_bits(state, nb_bits_out)
        state = self.coding_table[(state >> nb_bits_out) + symbol_tt['deltaFindState']]
        return state, eff

    def encode_efficient_data(self, inpu):
        """encode efficient data"""
        eff_list = []
        state, eff = self.encode_efficient(inpu[0], 0, self.symbol_tt)
        #eff_list.append(eff)
        for char in inpu:
            state, eff = self.encode_efficient(char, state, self.symbol_tt)
            eff_list.append(eff)
        eff = output_nb_bits(state - self.table_size, self.table_log) #Includes Current Bit
        eff_list.append(eff)
        bit_stream = ''.join(eff_list)
        return bit_stream


    def encode_symbol(self, symbol, state, bit_stream, symbol_tt):
        """Encode a Symbol Using tANS, giving the current state, the symbol, and the bitstream and STT"""
        symbol_tt = symbol_tt[symbol]
        nb_bits_out = (state + symbol_tt['deltaNbBits']) >> 16
        bit_stream += output_nb_bits(state, nb_bits_out)
        state = self.coding_table[(state >> nb_bits_out) + symbol_tt['deltaFindState']]
        return state, bit_stream

    def encode_data(self, inpu):
        """ Functions to Encode and Decode Streams of Data. """
        state, bit_stream = self.encode_symbol(inpu[0], 0, "", self.symbol_tt)
        bit_stream = ""
        for char in inpu:
            state, bit_stream = self.encode_symbol(char, state, bit_stream, self.symbol_tt)
        bit_stream += output_nb_bits(state - self.table_size, self.table_log) #Includes Current Bit
        return bit_stream

    def decode_data(self, bit_stream):
        """ decode data"""
        output = []
        state, bit_stream = bits_to_state(bit_stream, self.table_log)
        while len(bit_stream) > 0:
            symbol, state, bit_stream = decode_symbol(state, bit_stream, self.decode_table)
            output = [symbol] + output
        # cover a corner case when last symbols encoded with zero bits
        while self.decode_table[state]['nbBits'] == 0 and self.decode_table[state]['newX'] != 0:
            symbol, state, bit_stream = decode_symbol(state, bit_stream, self.decode_table)
            output = [symbol] + output
        return output

    @property
    def total_tables_size(self):
        """total tables size"""
        return len(self.coding_table) + 3 * len(self.decode_table) + 2 * len(self.symbol_tt)

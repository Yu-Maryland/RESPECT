from torch.utils.data import Dataset
import torch,random
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        #return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None
        y_ = d[:,:,1].view(d.shape[0],d.shape[1])
        #print(d.shape, list(y_.shape), ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).shape)
        idx = torch.Tensor([i for i in range(int(list(y_.shape)[1]))]).cuda().repeat(list(y_.shape)[0]).view(d.shape[0],d.shape[1])
        sorted, indices = torch.sort(y_, dim=1)
        #print(indices.shape, idx.shape)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cost = cos(indices.cuda(),idx).cuda()
        #print(cost.shape)
        return 1-cost, None 

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            #input, visited_dtype=torch.int64 if compress_mask else torch.uint8
            input, visited_dtype=torch.int64 if compress_mask else torch.bool
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    # 50, 1000000 
    def __init__(self, filename=None, size=25, num_samples=1000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            #self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]
            self.data = []
            for i in range(num_samples):
                graph = []
                for i in range(size):
                    graph.append([i,i+random.randint(1,10)])
                #print(torch.FloatTensor(graph).shape)
                self.data.append(torch.nn.functional.normalize(torch.FloatTensor(graph)))
 

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

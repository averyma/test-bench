class BinaryDataset(Dataset): 
    def __init__(self, sample_dim, num_samples, w):
        self.x = torch.zeros([num_samples, sample_dim])
        self.x[:,0].uniform_(-5,5)
        p = nn.Sigmoid()(self.x[:,0]*w)
#         self.y = ((self.x[:,0].sign()+1)/2.0).view(num_samples,1)\n",
        self.y = torch.stack([ B.Bernoulli(i).sample() for i in p ]).view(num_samples,1)
        self.num_samples = num_samples
    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx]

    def __len__(self):
        return self.num_samples

train_data = BinaryDataset(sample_dim = _sample_dim,\n",
                           num_samples = _num_samples_train,\n",
                           w = _w)\n",
test_data = BinaryDataset(sample_dim = _sample_dim, \n",
                          num_samples = _num_samples_test,\n",
                          w = _w)\n",
train_loader = DataLoader(dataset = train_data, \n",
                          batch_size = _batch_size, \n",
                          shuffle = True)\n",
test_loader = DataLoader(dataset = test_data, \n",
                         batch_size = _batch_size, \n",
                         shuffle = False)\n",

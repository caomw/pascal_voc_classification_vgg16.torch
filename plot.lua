require 'optim'
PATHS = require 'PATHS.lua'

loaded = torch.load(PATHS.LOG)

loaded.logger:style{['cost']='-', ['validation cost'] = '-'}
loaded.logger:plot('cost', 'validation cost')

print(loaded.mAPs)

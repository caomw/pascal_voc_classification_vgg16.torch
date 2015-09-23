local MultiClassNLLCriterion, parent = torch.class('MultiClassNLLCriterion', 'nn.Criterion')

function MultiClassNLLCriterion:__init()
   parent.__init(self)
   print('CAREFUL ! ALL TARGETS MUST BE EITHER -1 OR 1 !')
   self.sizeAverage=true
   
   self.sequence=nn.Sequential()
   self.sequence:add(nn.CMulTable())
   self.sequence:add(nn.MulConstant(-1,true))
   self.sequence:add(nn.SoftPlus())
   
   self.gradient=torch.Tensor()
end

function MultiClassNLLCriterion:updateOutput(input, target)
   self.sequence:forward({input,target})
   self.output=self.sequence.output:sum()
   if self.sizeAverage then self.output=self.output/input:size(1) end
   return self.output
end


function MultiClassNLLCriterion:updateGradInput(input, target)
   local p
   if self.sizeAverage then p=1/input:size(1) else p = 1 end
   self.gradient:resize(self.sequence.output:size()):fill(p)
   self.sequence:backward({input,target}, self.gradient)
   self.gradInput=self.sequence.gradInput[1]
   return self.gradInput
end

function MultiClassNLLCriterion:type(type)
   parent.type(self, type)
   self.sequence:type(type)
   self.gradient:type(type)
   return self
end

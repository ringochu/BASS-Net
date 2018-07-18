require 'nn'
require 'optim'
matio = require 'matio'
require 'xlua'
require 'pl'
require 'paths'
require 'torch'
require 'math'
require 'gnuplot'


local opt = lapp[[
   --path_dir         (default "./data/")            Path to the data directory
   -d, --data         (default "Indian_pines")       Dataset to use
   --development      (default 1)                    Use development dataset/ Whole training dataset
   -s,--save          (default "logs/")               subdirectory to save logs
   -p,--plot                                         plot while training
   -o,--optimization  (default "Adam")               optimization: SGD | LBFGS | Adam
   -l,--learningRate  (default 0.0005)              learning rate, for SGD only
   -b,--batchSize     (default 200)                  batch size
   -m,--momentum      (default 0)                    momentum, for SGD only
   -i,--maxIter       (default 1000)                 maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)                    L1 penalty on the weights
   --coefL2           (default 0)                    L2 penalty on the weights
   -t,--type          (default "cpu")                GPU or CPU
   --network          (default "MLP")                MLP or CNN
   --patch_size       (default 3)                    patch size of the image
   --nbands           (default 10)                   number of bands
   --block1_conv1     (default 3333)                 number of filters in the 1*1 convolution (here, 3333 is the sentinel value)
]]


if opt.development == 1 then
  opt.dev = ""
  if opt.data == "Salinas" then
    nval = 400
  else
    nval = 200
  end
else
  opt.dev = "_Full"
  nval = 0
end

if opt.data == "Salinas" then
  opt.nclasses = 16
else
  opt.nclasses = 9
end



-- Loading Data

-- Loading the testing set
test_data = matio.load(opt.path_dir .. opt.data .. "_Test_patch_" .. tostring(opt.patch_size) .. ".mat").test_patch
test_labels = matio.load(opt.path_dir ..opt.data .. "_Test_patch_" .. tostring(opt.patch_size) .. ".mat").test_labels:transpose(1,2)
opt.channels = test_data:size(2)
--Loading the training set
train_data = matio.load(opt.path_dir ..opt.data .. opt.dev .. "_Train_patch_" .. tostring(opt.patch_size) .. ".mat").train_patch:reshape(opt.nclasses*200-nval, opt.channels, opt.patch_size, opt.patch_size)
train_labels = matio.load(opt.path_dir ..opt.data .. opt.dev .. "_Train_patch_" .. tostring(opt.patch_size) .. ".mat").train_labels:transpose(1,2)
--Loading the validation set
val_data = matio.load(opt.path_dir ..opt.data .. "_Val_patch_" .. tostring(opt.patch_size) .. ".mat").val_patch
val_labels = matio.load(opt.path_dir ..opt.data .. "_Val_patch_" .. tostring(opt.patch_size) .. ".mat").val_labels:transpose(1, 2)

if opt.block1_conv1 == 3333 then
  opt.block1_conv1 = opt.channels
end

if opt.data ~= "PaviaU" then 
  while (opt.block1_conv1 % opt.nbands ~= 0) do
    opt.nbands = opt.nbands + 1
    print("Number of parallel networks reinitialized to " .. tostring(opt.nbands))
  end
end

trainset = {}
trainset.data = train_data
trainset.labels = train_labels

testset = {}
testset.data = test_data
testset.labels = test_labels

valset = {}
valset.data = val_data
valset.labels = val_labels

print("Trainset: ", trainset)
print("Testset: ", testset)
print("Valset: ", valset)

setmetatable(trainset, {__index = function(self, index)
             local input = self.data[index]
             local class = self.labels[index]
             local labelvector = torch.zeros(opt.nclasses)
             local label = labelvector
             label[class[1]+1] = 1
             local example = {input, label}
                                   return example
end})

setmetatable(testset, {__index = function(self, index)
             local input = self.data[index]
             local class = self.labels[index]
             local labelvector = torch.zeros(opt.nclasses)
             local label = labelvector
             label[class[1]+1] = 1
             local example = {input, label}
                                   return example
end})

setmetatable(valset, {__index = function(self, index)
             local input = self.data[index]
             local class = self.labels[index]
             local labelvector = torch.zeros(opt.nclasses)
             local label = labelvector
             label[class[1]+1] = 1
             local example = {input, label}
                                   return example
end})


function trainset:size()
    return trainset.data:size(1)
end
function testset:size()
    return testset.data:size(1)
end
function valset:size()
    return valset.data:size(1)
end
print("block1_conv1: " .. opt.block1_conv1)
opt.band_size = math.floor(opt.block1_conv1/opt.nbands) --TODO
print("band size: " .. opt.band_size)


------------------------- Start of BASS Net model ------------------------------------------------------------------------
model = nn.Sequential()

-- Block 2
model:add(nn.Reshape(opt.band_size, opt.patch_size*opt.patch_size))
-- nn.Temporalconvolution(inputFrameSize, outputFramesize, kW (kernel width of the conv), [dW] (step of conv, default as 1))
-- conv_xy - p,n
-- nn.TemporalConvolution(inputFrameSize (patch_size from input x patch_size from input), n, p, dW)

-- conv-3,20
model:add(nn.TemporalConvolution(opt.patch_size*opt.patch_size, 20, 3, 1))
model:add(nn.ReLU())

--conv-3,20
model:add(nn.TemporalConvolution(20, 20, 3, 1))
model:add(nn.ReLU())

-- conv-3,10
model:add(nn.TemporalConvolution(20, 15, 3, 1))
model:add(nn.ReLU())

--conv-5,5
model:add(nn.TemporalConvolution(15, 5, 5, 1))
model:add(nn.ReLU())

model:add(nn.Reshape((opt.band_size)*5, 1)) 


parallel_model = nn.Parallel(2, 2)

for i = 1, opt.nbands do
    parallel_model:add(model:clone())
end


-- net is the main model for the whole BASS architecture
net = nn.Sequential()
-- nn.SpatialConvolution(nInputPlane, nOutputPlane, kW(), kH(kernel height of conv))
-- IP: (220, 220, 1, 1); Salinas: (224, 224, 1, 1); PaviaU: (103, 103, 1, 1)
print('nInputPlane, nOutputPlane: ' .. opt.channels .. ', ' .. opt.block1_conv1)
--Block 1
net:add(nn.SpatialConvolution(opt.channels, opt.block1_conv1, 1, 1))
net:add(nn.ReLU()) 
net:add(nn.Reshape(opt.nbands, opt.band_size, opt.patch_size*opt.patch_size))

--print('10, 22, 3x3 ' .. opt.nbands .. ', ' .. opt.band_size .. ', ' ..opt.patch_size*opt.patch_size )

-- Adding in block 2
net:add(parallel_model)
net:add(nn.Reshape(opt.nbands*(opt.band_size)*5))

--nbands: Ip: 10; Salinas: 14; PaviaU: 1
--band_size:  IP: 22; Salinas: 16; PaviaU:103
-- Linear(inputSize, outputSize, bias)

--Block 3
net:add(nn.Linear(opt.nbands*(opt.band_size-10)*5, 100)) --fc-100; IP: (600, 100)
--print('net weight: ')
--print(net.weight)
net:add(nn.ReLU())
net:add(nn.Dropout())
net:add(nn.Linear(100, opt.nclasses)) -- This is fc-9

net:add(nn.LogSoftMax())


-- Parameter Sharing
parallel_model = net:get(4)  
for band = 2, opt.nbands do
  local current_module = parallel_model:get(band)
  current_module:share(parallel_model:get(1), 'weight', 'bias',
                       'gradWeight', 'gradBias')
end

net:training()
--Using loss criterion as we are doing classification with more than 2 classes and including nn.LogSoftMax at the end of the network
criterion = nn.ClassNLLCriterion() 

if opt.data == "Salinas" then
  classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
else
  classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8'}
end

-- getParameters return 2 tensors:
-- parameters: flattened learnable parameters (weights)
-- gradParameters: gradients of the energy wrt the learnable parameters
parameters, gradParameters = net:getParameters()
local trained_data = assert(io.open('trained_data.txt','wb'))
trained_data:write('Weights: \n', tostring(parameters), '\n\n\n', 'Grad weights: \n', tostring(gradParameters), '\n')
confusion = optim.ConfusionMatrix(classes)

trainLogger = optim.Logger(paths.concat("./" .. opt.save .. opt.data, 'train.log'))
valLogger = optim.Logger(paths.concat("./" .. opt.save .. opt.data, 'val.log'))
testLogger = optim.Logger(paths.concat("./" .. opt.save .. opt.data, 'test.log'))








---------------------------- Statistics ----------------------------------------
-- This function is for finding the recall, precision and F1 score 
function stat(con_mat)
  diag = 0
  for i=1, opt.nclasses do
    diag = diag + con_mat[{{i, i}, {i, i}}]
  end


  for i=1, opt.nclasses do
    local tp = (con_mat[{{i,i},{i,i}}])
    local fp = (con_mat[{{1,opt.nclasses},{i,i}}])
    local fn = (con_mat[{{i,i},{1,opt.nclasses}}])
    
    local tp_sum = tp:sum()
    local fp_sum = fp:sum() - tp_sum
    local fn_sum = fn:sum() - tp_sum
    local tn = diag:sum() - tp_sum
    print('Statistics for class ' .. i)
    print('tp, fp, fn, tn: ' .. tostring(tp_sum) .. ' ' .. tostring(fp_sum) .. ' ' .. tostring(fn_sum) .. ' ' .. tostring(tn))

    precision = (tp_sum / (fp_sum + tp_sum))* 100
    if precision ~= precision then -- when divided by 0
      precision = 0
    end
    print('Precision: ' .. precision .. '%')
    recall = (tp_sum /(fn_sum + tp_sum)) * 100
    print('Recall: ' .. recall .. '%')

    f1 = 2*(precision*recall)/(precision+recall)
    if f1 ~= f1 then f1 = 0 end
    print('F1 score: ' .. f1 .. '% \n')


  end 

end
------------------------- TRAIN Func --------------------------------------------
train_oa = {}
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
 
   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      
      -- created minibatch of 200 
      local inputs = torch.Tensor(opt.batchSize, opt.channels, opt.patch_size, opt.patch_size)
      local targets = torch.Tensor(opt.batchSize)

      if opt.type == "cuda" then
      	inputs:cuda()  --Cuda(compute Unified Device Architecture)
      	targets:cuda()
      end
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = net:forward(inputs) --Send the batch of inputs to the model and get all the outputs in a single run
         local f = criterion:forward(outputs, targets) --loss. To evaluate how wrong the model is

         -- estimate df/dW
	 -- propagate the error back to the model and ask the model to calculate how much each neuron needs to change to eliminate the error
         local df_do = criterion:backward(outputs, targets) --dloss_doutputs
         net:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         -- print(inputs:size(1))
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end
         -- return f and df/dX
         --print('loss: ' .. tostring(f)) 
         return f,gradParameters
    end

      -- optimize on current mini-batch
      -- Learning Rate secheduling??
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)

         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD(Stochastic Gradient Descent) step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      elseif opt.optimization == "Adam" then

      	 adamState = adamState or {
      	 	learningRate = opt.learningRate,
      	 	momentum = opt.momentum,
      	 	learningRateDecay = 5e-9
      	 }
      	 optim.adam(feval, parameters, adamState)
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()

   -- print confusion matrix
   print(confusion)
   print('% mean class accuracy (train set)' .. tostring(confusion.totalValid*100)) 
   print('Time taken: '.. tostring(time))
   table.insert(train_oa, confusion.totalValid*100)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()
   epoch = epoch + 1
  
   return (1 - confusion.totalValid)*100
end


-------------------------------- val func ------------------------------------------------------
best_val = 0
early_stop_counter = 0
val_oa = {}
function val(dataset)

    net:evaluate() -- This sets the mode of the Modeule to train=false
    local time = sys.clock()

    for t = 1, dataset:size(), opt.batchSize do
        xlua.progress(t, dataset:size())
        local inputs = torch.Tensor(opt.batchSize, opt.channels, opt.patch_size, opt.patch_size)
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t, math.min(t+opt.batchSize-1, dataset:size()) do
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k+1
        end

        local preds = net:forward(inputs) -- Computing Loss
        --print(preds)
        for l = 1, k - 1 do
            confusion:add(preds[l], targets[l])
        end

    end
    print(confusion)
    local con_mat = confusion.mat
    stat(con_mat)
    if confusion.totalValid > best_val 
      then
        early_stop_counter = 0
        best_val = confusion.totalValid
      else 
        early_stop_counter = early_stop_counter + 1
    end
    print("Best validation accuracy yet :" .. tostring(best_val*100) .. "%")
    table.insert(val_oa, confusion.totalValid*100)
    valLogger:add{['% mean class accuracy (val set)'] = confusion.totalValid * 100}
    confusion:zero()

    net:training()
    

end

------------------------------ Test Func ------------------------------------------------------
best_test = 0
test_oa ={}
function test(dataset)

    net:evaluate()
    
    --default batchSize:200, the batchSize as the step to increment
    for t = 1, dataset:size(), opt.batchSize do
        xlua.progress(t, dataset:size())
        local inputs = torch.Tensor(opt.batchSize, opt.channels, opt.patch_size, opt.patch_size)
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t, math.min(t+opt.batchSize-1, dataset:size()) do
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k+1
        end

      local preds = net:forward(inputs) -- Computing Loss
        for l = 1, k - 1 do
            confusion:add(preds[l], targets[l])
        end
    end
    print(confusion)
    local con_mat = confusion.mat
    stat(con_mat)

    if confusion.totalValid > best_test then
        best_test = confusion.totalValid
        torch.save("./pretrained/final_Block_2_no_fc.t7", net)
    end
    print("Best test accuracy yet :" .. tostring(best_test*100) .. "%")
    table.insert(test_oa, confusion.totalValid*100)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero() 
    net:training()

end


-------------------------------- MAIN----------------------------------
local time_main = sys.clock()
for i = 1, opt.maxIter do
    print("counter: ", early_stop_counter)
    if early_stop_counter <= 100 then
        ------ check time for each layer-----
--       net:replace(function(module) return nn.Profile(module, 1) end)

      train(trainset)
      if opt.development == 1 then
         val(valset)
      else
         test(testset)
      end
     else
       print('Early stopped, accuracy stopped increasing at: ', i)
       break
    end
end
print('Training time for whole process: ' .. tostring(sys.clock() - time_main))


  if opt.development == 1 then 
      Accuracy = torch.Tensor{train_oa, val_oa}
      gnuplot.plot({'Training', Accuracy[1]},{'Validation', Accuracy[2]})
    else 
      Accuracy = torch.Tensor{train_oa, test_oa}
      gnuplot.plot({'Training', Accuracy[1]},{'Testing', Accuracy[2]})
  end

gnuplot.xlabel('epoch')
gnuplot.ylabel('overall accuracy')   
gnuplot.closeall()
trained_data:close()

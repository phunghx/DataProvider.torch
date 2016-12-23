local LMDBProvider = torch.class('DataProvider.LMDBProvider')
---local dbg = require("debugger")

function LMDBProvider:__init(...)
    xlua.require('torch',true)
    require 'lmdb'
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataProvider ',
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', req = true},
    {arg='Source', type='userdata', help='LMDB env', req=true},
    {arg='Verbose', type='boolean', help='display messages', default = false},
    {arg='outputSize', type='userdata', help='size of data', default = {3,224,224}},
    {arg='getKey',type='function', help='function used to get key', req=false},
    {arg='skipFrame',type='int', help='skip frame in video', default=3}

    )
    for x,val in pairs(args) do
        self[x] = val
    end
    self.Config = ...
end

function LMDBProvider:size()
    self.Source:open()
    local SizeData = self.Source:stat()['entries']
    self.Source:close()
    return SizeData
end

function LMDBProvider:cacheSeq(start_pos, num, data, labels,train)
    
    local time = 0
    if #self.outputSize == 4 then
    	time = self.outputSize[2]
    end
    local num = num or 1
    self.Source:open()
    local txn = self.Source:txn(true)
    local cursor = txn:cursor()
    cursor:set(start_pos)

    local Data = data or {}
    local Labels = labels or {}
    if time == 0 then
        for i = 1, num do
	     local key, data = cursor:get()
	     Data[i], Labels[i] = self.ExtractFunction(data, key)
	     if i<num then
		    cursor:next()
	     end
	end
    else
    	start_index = tonumber(start_pos)
        for i = 1, num do
             start_cursor = start_index- self.skipFrame * time + torch.randperm(self.skipFrame)[1] - 1 + i
             start_data = self.getKey(start_cursor)
             cursor:set(start_data)
             ---if train==false then
	     ---        dbg()
	     ---end
             local Data_t = torch.ByteTensor(self.outputSize[2],self.outputSize[1],self.outputSize[3],self.outputSize[4])
             Label_t = nil
             for t=1, time do
		     local key, data = cursor:get()
		     ---if data == nil then
		     --	dbg()
		     ---end
		     Data_t[t], Label_t = self.ExtractFunction(data, key)
		     if t<time then
		     	    for tem_index=1,self.skipFrame do
				cursor:next()
			    end
		     end
	     end
	     Data[i], Labels[i] = Data_t:transpose(1,2), Label_t
	end
    end
    cursor:close()
    txn:abort()
    self.Source:close()
    return Data, Labels
end

function LMDBProvider:cacheRand(keys, data, labels)
    local num
    if type(keys) == 'table' then
        num = #keys
    else
        num = keys:size(1)
    end
    self.Source:open()
    local txn = self.Source:txn(true)
    local Data = data or {}
    local Labels = labels or {}

    for i = 1, num do
        local item = txn:get(keys[i])
        Data[i], Labels[i] = self.ExtractFunction(item, keys[i])
    end
    txn:abort()
    self.Source:close()
    return Data, Labels
end

function LMDBProvider:threads(nthread)
    local nthread  = nthread or 1
    local config = self.Config
    local threads = require 'threads'
    threads.serialization('threads.sharedserialize')
    self.threads = threads(nthread,
    function()
        require 'lmdb'
        local DataProvider = require 'DataProvider'
    end,
    function(idx)
        workerProvider = DataProvider.LMDBProvider(config)
        lmdb.verbose = config.Verbose
    end
    )
end

function LMDBProvider:asyncCacheSeq(start, num, dataBuffer, labelsBuffer,train)
    self.threads:addjob(
    -- the job callback (runs in data-worker thread)
    function()
        local data, labels = workerProvider:cacheSeq(start,num,dataBuffer,labelsBuffer,train)
        return data, labels
    end,
    -- the endcallback (runs in the main thread)
    function(data,labels)
    end
    )
end

function LMDBProvider:asyncCacheRand(keys, dataBuffer, labelsBuffer)
    self.threads:addjob(
    -- the job callback (runs in data-worker thread)
    function()
        local data, labels = workerProvider:cacheRand(keys,dataBuffer,labelsBuffer)
        return data, labels
    end,
    -- the endcallback (runs in the main thread)
    function(data,labels)
    end
    )
end

function LMDBProvider:synchronize()
    return self.threads:synchronize()
end

#ifndef SYSU_STACK_H_
#define SYSU_STACK_H_

class SysuStack
{
private:
	void **items;
	size_t max_size;
	int current;
public:
	SysuStack(const size_t &max_size = 32):
		items(new void*[max_size]),
		max_size(max_size),
		current(-1)
	{}
	
	~SysuStack()
	{
		delete items;
	}
	
	bool push(void *item)
	{
		if(current < max_size)
			items[++current] = item;
		return current < max_size;
	}
	
	void *top()
	{
		if(current > -1)
			return items[current];
		else
			return NULL;
	}
	
	bool pop()
	{
		if(current > -1)
		{
			current--;
			return true;
		}
		else
			return false;
	}
	
	size_t size()
	{
		return size_t(current + 1);
	}
	
	size_t maxSize()
	{
		return max_size;
	}
	
	bool empty()
	{
		return current == -1;
	}
};

#endif /*  SYSU_STACK_H_ */

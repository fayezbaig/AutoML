*I2C1_CR1 |= (1 << 0);   /* Enable peripheral */

	volatile uint32_t* I2C1_SR1 = (volatile uint32_t*)(I2C1_BASE_ADDR + 0x14U);
	volatile uint32_t* I2C1_SR2 = (volatile uint32_t*)(I2C1_BASE_ADDR + 0x18U);
	volatile uint32_t* I2C1_DR  = (volatile uint32_t*)(I2C1_BASE_ADDR + 0x10U);
	volatile uint32_t  tmp;

    /* wait until DR is empty (TXE), then write */
    #define SEND(b)  do { while (!(*I2C1_SR1 & (1U << 7))); *I2C1_DR = (b); } while (0)

	/* ---------- Transaction 1: commands ---------- */
	*I2C1_CR1 |= (1 << 8);                    /* START */
	while (!(*I2C1_SR1 & (1U << 0)));         /* wait SB */
	*I2C1_DR = 0x78U;                         /* address */
	while (!(*I2C1_SR1 & (1U << 1)));         /* wait ADDR (ACK) */
	tmp = *I2C1_SR1; tmp = *I2C1_SR2;         /* clear ADDR */

	SEND(0x00);                               /* following bytes are commands */
	SEND(0xAE);                               /* display off */
	SEND(0x8D); SEND(0x14);                   /* charge pump on */
	SEND(0x20); SEND(0x00);                   /* horizontal addressing */
	SEND(0x21); SEND(0x00); SEND(0x7F);       /* columns 0..127 */
	SEND(0x22); SEND(0x00); SEND(0x07);       /* pages 0..7 */
	SEND(0xAF);                               /* display on */

	while (!(*I2C1_SR1 & (1U << 2)));         /* wait BTF: last byte sent */
	*I2C1_CR1 |= (1 << 9);                    /* STOP */

	/* ---------- Transaction 2: pixel data ---------- */
	while (*I2C1_SR2 & (1U << 1));            /* wait bus free */
	*I2C1_CR1 |= (1 << 8);                    /* START */
	while (!(*I2C1_SR1 & (1U << 0)));
	*I2C1_DR = 0x78U;
	while (!(*I2C1_SR1 & (1U << 1)));
	tmp = *I2C1_SR1; tmp = *I2C1_SR2;

	SEND(0x40);                               /* following bytes are pixels */

	/* 'H' at top-left: 5 columns + 1 blank */
	SEND(0x7F); SEND(0x08); SEND(0x08); SEND(0x08); SEND(0x7F); SEND(0x00);

	for (int i = 6; i < 1024; i++)            /* clear the rest of the screen */
	    SEND(0x00);

	while (!(*I2C1_SR1 & (1U << 2)));
	*I2C1_CR1 |= (1 << 9);                    /* STOP */

	(void)tmp;
	for(;;);

}
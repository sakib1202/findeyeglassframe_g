// Array of frame names and prices
const frames = [
    { name: "Stylish Round Frame", price: 50, image: "frame1.jpg", arModel: "frame1.gltf" },
    { name: "Classic Oval Frame", price: 55, image: "frame2.jpg", arModel: "frame2.gltf" },
    { name: "Square Frame", price: 60, image: "frame3.jpg", arModel: "frame3.gltf" },
    { name: "Bold Black Frame", price: 65, image: "frame4.jpg", arModel: "frame4.gltf" },
    { name: "Retro Cat-Eye Frame", price: 70, image: "frame5.jpg", arModel: "frame5.gltf" },
    { name: "Modern Metal Frame", price: 75, image: "frame6.jpg", arModel: "frame6.gltf" },
    { name: "Vintage Wooden Frame", price: 80, image: "frame7.jpg", arModel: "frame7.gltf" },
    { name: "Sleek Rectangle Frame", price: 85, image: "frame8.jpg", arModel: "frame8.gltf" },
    { name: "Sporty Aviator Frame", price: 90, image: "frame9.jpg", arModel: "frame9.gltf" },
    { name: "Chic Butterfly Frame", price: 95, image: "frame10.jpg", arModel: "frame10.gltf" },
    { name: "Bold Oversized Frame", price: 100, image: "frame11.jpg", arModel: "frame11.gltf" },
    { name: "Elegant Oval Frame", price: 105, image: "frame12.jpg", arModel: "frame12.gltf" },
    { name: "Minimalist Thin Frame", price: 110, image: "frame13.jpg", arModel: "frame13.gltf" },
    { name: "Luxurious Gold Frame", price: 115, image: "frame14.jpg", arModel: "frame14.gltf" },
    { name: "Casual Black Frame", price: 120, image: "frame15.jpg", arModel: "frame15.gltf" },
    { name: "Hipster Round Frame", price: 125, image: "frame16.jpg", arModel: "frame16.gltf" },
    { name: "Cool Neon Frame", price: 130, image: "frame17.jpg", arModel: "frame17.gltf" },
    { name: "Classic Blue Frame", price: 135, image: "frame18.jpg", arModel: "frame18.gltf" },
    { name: "Smart Green Frame", price: 140, image: "frame19.jpg", arModel: "frame19.gltf" },
    { name: "Elegant Red Frame", price: 145, image: "frame20.jpg", arModel: "frame20.gltf" },
    { name: "Funky Transparent Frame", price: 150, image: "frame21.jpg", arModel: "frame21.gltf" },
    { name: "Rich Purple Frame", price: 155, image: "frame22.jpg", arModel: "frame22.gltf" },
    { name: "Glamorous White Frame", price: 160, image: "frame23.jpg", arModel: "frame23.gltf" },
    { name: "Sleek Silver Frame", price: 165, image: "frame24.jpg", arModel: "frame24.gltf" },
    { name: "Charming Pink Frame", price: 170, image: "frame25.jpg", arModel: "frame25.gltf" },
    { name: "Warm Brown Frame", price: 175, image: "frame26.jpg", arModel: "frame26.gltf" },
    { name: "Bold Orange Frame", price: 180, image: "frame27.jpg", arModel: "frame27.gltf" },
    { name: "Trendy Yellow Frame", price: 185, image: "frame28.jpg", arModel: "frame28.gltf" },
    { name: "Mysterious Black Frame", price: 190, image: "frame29.jpg", arModel: "frame29.gltf" },
    { name: "Sophisticated Gray Frame", price: 195, image: "frame30.jpg", arModel: "frame30.gltf" },
  ];
  
  // Function to render the product cards dynamically
  function renderProducts() {
    const productList = document.getElementById('product-list');
  
    frames.forEach(frame => {
      const productCard = document.createElement('div');
      productCard.classList.add('product-card');
  
      productCard.innerHTML = `
        <img src="${frame.image}" alt="${frame.name}">
        <h3>${frame.name}</h3>
        <p class="price">$${frame.price}</p>
        <button class="add-to-cart" data-name="${frame.name}" data-price="${frame.price}">Add to Cart</button>
        <button class="view-ar" data-ar="${frame.arModel}">Try in AR</button>
      `;
  
      productList.appendChild(productCard);
    });
  }
  
  // Call the renderProducts function to load the product cards when the page is loaded
  renderProducts();
  
  // AR Viewer toggle functionality (same as before)
  const arView = document.getElementById('ar-view');
  const arFrameAsset = document.getElementById('ar-frame');
  const closeArButton = document.getElementById('close-ar');
  const viewArButtons = document.querySelectorAll('.view-ar');
  
  viewArButtons.forEach(button => {
    button.addEventListener('click', (event) => {
      const arModelSrc = event.target.getAttribute('data-ar');
      arView.style.display = 'block';
      arFrameAsset.setAttribute('src', arModelSrc);
    });
  });
  
  closeArButton.addEventListener('click', () => {
    arView.style.display = 'none';
  });
  